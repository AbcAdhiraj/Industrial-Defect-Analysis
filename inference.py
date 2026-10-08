"""Module 3 - inference engine: forward chaining, backward chaining, explanation.

Facts are a dict {attribute: value}. Base facts are discretised features
({"Y_Perimeter": "LOW", ...}); derived facts are symptoms ({"ELONGATED_X": "YES"});
the final conclusion is {"fault": "<class>"}.
"""
from collections import namedtuple

from rules import RULES, DERIVED

FCResult = namedtuple("FCResult", "label winner fired conflicts facts")
Proof = namedtuple("Proof", "goal rule children")   # rule None => leaf fact


def holds(condition, facts):
    """True if the fact store satisfies one condition. O(1)."""
    attr, allowed = condition
    return facts.get(attr) in allowed


def applicable(rule, facts):
    """All conditions satisfied (an empty condition list is always true). O(C)."""
    return all(holds(c, facts) for c in rule.conditions)


def forward_chain(base_facts, rules=RULES):
    """Data-driven inference. Returns FCResult(label, winner, fired, conflicts, facts).

    Purpose: from the measured facts derive every symptom, collect all fault
    rules that apply, then resolve the conflict between them.
    Phase 1 (saturation): repeat passes over the symptom rules, adding derived
    facts, until a pass adds nothing new. Phase 2 (conflict resolution): among
    applicable fault rules pick the max of (priority, specificity=#conditions,
    earlier rule wins ties). Time O(P * R * C) with P passes (P <= depth of the
    symptom chain, here 1-2), space O(R).
    WHY it works: symptom rules only ever ADD facts (monotonic), so the
    fact set can only grow and is bounded -> the loop terminates, and the
    order in which rules are tried cannot change the final fact set.
    """
    facts, fired = dict(base_facts), []
    symptom_rules = [r for r in rules if r.conclusion[0] != "fault"]
    changed = True
    while changed:
        changed = False
        for rule in symptom_rules:
            if rule not in fired and applicable(rule, facts):
                facts[rule.conclusion[0]] = rule.conclusion[1]
                fired.append(rule)
                changed = True
    candidates = [r for r in rules if r.conclusion[0] == "fault" and applicable(r, facts)]
    fired += candidates
    order = {r.rid: i for i, r in enumerate(rules)}
    winner = max(candidates, key=lambda r: (r.priority, len(r.conditions), -order[r.rid]))
    conflicts = [r for r in candidates if r.conclusion[1] != winner.conclusion[1]]
    facts["fault"] = winner.conclusion[1]
    return FCResult(winner.conclusion[1], winner, fired, conflicts, facts)


def classify(features, thresholds, rules=RULES):
    """Numeric features -> fault label (discretise, then forward-chain). O(F + R*C)."""
    return forward_chain(thresholds.discretise(features), rules).label


def backward_chain(goal, facts, rules=RULES, _path=frozenset()):
    """Goal-driven proof search. Returns a Proof tree or None ("not provable").

    Purpose: answer "can the goal (attr, value) be shown from these facts,
    and by which rules?" Depth-first over rules that conclude the goal; for
    each, prove every condition (symptom conditions recursively, base-feature
    conditions by looking them up). Time O(R*C) per call on an acyclic rule
    base (each goal is expanded at most once per path), space O(depth).
    WHY it works: a goal holds iff some rule concluding it has all its
    conditions provable - the same truth the forward chainer computes, found
    from the goal downwards. `_path` stops infinite loops on cyclic rules.
    NOTE: proving "fault == X" shows X is SUPPORTED; X may still lose the
    conflict resolution that forward chaining performs.
    """
    attr, value = goal
    if facts.get(attr) == value:
        return Proof(goal, None, [])
    if goal in _path:
        return None
    for rule in sorted((r for r in rules if r.conclusion == goal), key=lambda r: -r.priority):
        children = []
        for cond_attr, allowed in rule.conditions:
            if cond_attr in DERIVED:
                child = backward_chain((cond_attr, "YES"), facts, rules, _path | {goal})
            elif facts.get(cond_attr) in allowed:
                child = Proof((cond_attr, facts[cond_attr]), None, [])
            else:
                child = None
            if child is None:
                break
            children.append(child)
        else:
            return Proof(goal, rule, children)
    return None


def facts_needed(goal, rules=RULES):
    """For each rule that could conclude `goal`, the base-feature conditions
    needed (symptom conditions are expanded recursively). O(R*C)."""
    out = []
    for rule in (r for r in rules if r.conclusion == goal):
        needed = []
        for attr, allowed in rule.conditions:
            if attr in DERIVED:
                for sub in facts_needed((attr, "YES"), rules):
                    needed += sub[1]
            else:
                needed.append((attr, sorted(allowed)))
        out.append((rule.rid, needed))
    return out


def format_proof(proof, indent=0):
    """Indented text rendering of a proof tree."""
    pad = "  " * indent
    attr, value = proof.goal
    if proof.rule is None:
        return pad + "%s = %s   [fact]" % (attr, value)
    lines = [pad + "%s = %s   [rule %s: %s]" % (attr, value, proof.rule.rid, proof.rule.text)]
    return "\n".join(lines + [format_proof(c, indent + 1) for c in proof.children])


def explain(features, thresholds, rules=RULES):
    """Human-readable trace: facts used, rules fired, conflicts resolved. O(F + R*C)."""
    facts = thresholds.discretise(features)
    result = forward_chain(facts, rules)
    used = sorted({a for r in result.fired for a, _ in r.conditions if a not in DERIVED})
    lines = ["Facts used (discretised): " + ", ".join("%s=%s" % (a, facts[a]) for a in used),
             "Rules fired (%d):" % len(result.fired)]
    lines += ["  [%s] p=%d  %s" % (r.rid, r.priority, r.text) for r in result.fired]
    lines.append("Conflict resolution: " + (
        "%d rule(s) concluded other faults (%s); winner chosen by priority, then specificity"
        % (len(result.conflicts), ", ".join("%s->%s" % (r.rid, r.conclusion[1])
                                            for r in result.conflicts))
        if result.conflicts else "none (all applicable fault rules agree)"))
    w = result.winner
    lines.append("Decision: %s by rule %s (priority %d, %d conditions)"
                 % (result.label, w.rid, w.priority, len(w.conditions)))
    proof = backward_chain(("fault", result.label), facts, rules)
    lines.append("Proof (backward chaining):\n" + (format_proof(proof, 1) if proof else "  none"))
    return "\n".join(lines)
