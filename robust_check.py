"""Module 2 - adversarial search: is the expert system's answer stable under noise?

MODELLING CHOICE (this is a design decision for the project, not a fact of
the data): we treat robustness checking as a two-player zero-sum game.
  MAX = the inspector.      MIN = worst-case sensor noise.
Borderline features = features used by the fired rules that lie within a noise
band `delta` of a discretisation threshold (at most 4, closest first).
  MAX moves: TRUST, REVIEW, REMEASURE(f)  (REMEASURE halves f's band, cost 0.2,
             each feature at most once, only while f has not been perturbed).
  MIN moves: PERTURB(f, p) with p in {-1, 0, +1}: reading of f becomes
             nominal + p * band(f), for a not-yet-fixed borderline feature.
Turn order: MAX, MIN, MAX, MIN, ... TRUST ends MAX's turns: MIN then fixes all
remaining features one by one (these closing plies do not count against the
depth limit) and the game ends. REVIEW ends the game at once.
Payoffs (kept as integers in tenths to avoid float drift): REVIEW = -0.5;
TRUST = +1 if the expert-system label under the perturbed readings equals the
nominal label, else -2; every REMEASURE adds -0.2.
Depth limit D (default 4 plies) applies to the MAX/MIN alternation; a node cut
off at the limit is scored cost + REVIEW, the value MAX can always secure
(a conservative static evaluation). True labels are NEVER used in the game.
"""
from collections import namedtuple

from inference import classify, forward_chain
from rules import DERIVED

REVIEW_PAY, TRUST_OK, TRUST_BAD, REMEASURE_COST = -5, 10, -20, -2   # tenths
INF = float("inf")


# ------------------------------------------------------------- generic solvers
def minimax(game, state, stats):
    """Plain minimax. Returns (value, best_move); counts visited nodes in stats[0].

    Purpose: the value of a position when both sides play optimally.
    Time O(b^d) (b = branching, d = depth), space O(d) for the recursion.
    WHY it works: MAX picks the child with the largest value, MIN the smallest;
    by induction from the leaves every node's value is what optimal play
    achieves from there. Ties keep the FIRST best move (strict '>' / '<').
    """
    stats[0] += 1
    if game.is_terminal(state):
        return game.utility(state), None
    if game.cut(state):
        return game.evaluate(state), None
    maximizing = game.is_max(state)
    best, best_move = (-INF if maximizing else INF), None
    for move in game.moves(state):
        value, _ = minimax(game, game.result(state, move), stats)
        if (value > best) if maximizing else (value < best):
            best, best_move = value, move
    return best, best_move


def alphabeta(game, state, stats, alpha=-INF, beta=INF):
    """Minimax with alpha-beta pruning: same value and same best move, fewer nodes.

    alpha = best value MAX can already guarantee on the path, beta = best MIN
    can guarantee. Time O(b^(d/2)) with perfect move ordering, O(b^d) worst
    case; space O(d). WHY it is safe: once a MIN node's value drops to <= alpha,
    MAX (who already has alpha elsewhere) will never choose this branch, so its
    remaining children cannot change the result; symmetric for MAX with beta.
    Pruned values are bounds, but a bound never beats the strictly better
    value that caused the prune, so the root value and first-best move agree.
    """
    stats[0] += 1
    if game.is_terminal(state):
        return game.utility(state), None
    if game.cut(state):
        return game.evaluate(state), None
    maximizing = game.is_max(state)
    best, best_move = (-INF if maximizing else INF), None
    for move in game.moves(state):
        value, _ = alphabeta(game, game.result(state, move), stats, alpha, beta)
        if maximizing:
            if value > best:
                best, best_move = value, move
            alpha = max(alpha, best)
        else:
            if value < best:
                best, best_move = value, move
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best, best_move


class TreeGame:
    """Tiny explicit game tree (nested lists, ints at leaves) for tests/teaching.
    State = (subtree, max_to_move)."""

    def __init__(self, tree):
        self.tree = tree

    def initial(self):
        return (self.tree, True)

    def is_terminal(self, s): return not isinstance(s[0], list)
    def utility(self, s): return s[0]
    def cut(self, s): return False
    def evaluate(self, s): return 0
    def is_max(self, s): return s[1]
    def moves(self, s): return range(len(s[0]))
    def result(self, s, m): return (s[0][m], not s[1])


def random_tree(rng, depth, branching):
    """Random game tree of fixed depth; leaves are integers in [-9, 9]."""
    if depth == 0:
        return rng.randint(-9, 9)
    return [random_tree(rng, depth - 1, rng.randint(1, branching)) for _ in range(rng.randint(1, branching))]


# ----------------------------------------------------------- the noise game
class RobustnessGame:
    """Concrete game; state = (ply, fixed, halved, cost, committed, reviewed)."""

    def __init__(self, features, thresholds, borderline, depth_limit=4):
        self.features, self.thresholds = features, thresholds
        self.names = [name for name, _ in borderline]
        self.deltas = [delta for _, delta in borderline]
        self.depth_limit = depth_limit
        self.nominal = classify(features, thresholds)
        self._label_cache = {}

    def initial(self):
        n = len(self.names)
        return (0, (None,) * n, (False,) * n, 0, False, False)

    def is_terminal(self, s):
        ply, fixed, halved, cost, committed, reviewed = s
        return reviewed or (committed and None not in fixed)

    def is_max(self, s):
        return not s[4] and s[0] % 2 == 0

    def cut(self, s):
        return not s[4] and s[0] >= self.depth_limit

    def evaluate(self, s):
        return s[3] + REVIEW_PAY

    def _label(self, s):
        """Expert-system label under the offsets in state s (memoised). The
        offsets alone determine the label, so equal offsets share one call."""
        _, fixed, halved, _, _, _ = s
        offsets = tuple(0.0 if p is None else p * d / (2 if h else 1)
                        for p, d, h in zip(fixed, self.deltas, halved))
        if offsets not in self._label_cache:
            changed = dict(self.features)
            for name, off in zip(self.names, offsets):
                changed[name] += off
            self._label_cache[offsets] = classify(changed, self.thresholds)
        return self._label_cache[offsets]

    def utility(self, s):
        if s[5]:
            return s[3] + REVIEW_PAY
        return s[3] + (TRUST_OK if self._label(s) == self.nominal else TRUST_BAD)

    def moves(self, s):
        ply, fixed, halved, cost, committed, reviewed = s
        free = [i for i, p in enumerate(fixed) if p is None]
        if self.is_max(s):
            return ["TRUST", "REVIEW"] + [("REMEASURE", i) for i in free if not halved[i]]
        return [("PERTURB", i, p) for i in free for p in (-1, 1, 0)] or ["PASS"]

    def result(self, s, move):
        ply, fixed, halved, cost, committed, reviewed = s
        if move == "TRUST":
            return (ply + 1, fixed, halved, cost, True, False)
        if move == "REVIEW":
            return (ply + 1, fixed, halved, cost, committed, True)
        if move == "PASS":
            return (ply + 1, fixed, halved, cost, committed, False)
        if move[0] == "REMEASURE":
            new = tuple(h or i == move[1] for i, h in enumerate(halved))
            return (ply + 1, fixed, new, cost + REMEASURE_COST, committed, False)
        new = tuple(move[2] if i == move[1] else p for i, p in enumerate(fixed))
        return (ply + 1, new, halved, cost, committed, False)

    def move_name(self, move):
        if isinstance(move, tuple):
            return "%s(%s)" % (move[0], self.names[move[1]])
        return move


def borderline_features(features, thresholds, fired_rules, noise_frac=0.10, limit=4):
    """Features of fired rules within a noise band of a threshold.

    Purpose: shrink the game to the few readings whose noise could matter.
    For each measured (non-derived, non-binary) feature used by a fired rule,
    distance = |value - nearest threshold|; keep it if distance <= delta where
    delta = noise_frac * train std. Closest (relative to delta) first, at most
    `limit`. Time O(R*C). Limitation: rules that did NOT fire are ignored, so a
    reading that could switch ON a new rule is not considered.
    """
    names = {a for r in fired_rules for a, _ in r.conditions if a not in DERIVED}
    found = []
    for name in sorted(names):
        delta = thresholds.noise_delta(name, noise_frac)
        if delta <= 0:
            continue
        low, high, _ = thresholds.table[name]
        dist = min(abs(features[name] - low), abs(features[name] - high))
        if dist <= delta:
            found.append((dist / delta, name, delta))
    return [(name, delta) for _, name, delta in sorted(found)[:limit]]


RobustResult = namedtuple("RobustResult", "decision value move nominal borderline "
                                          "minimax_nodes alphabeta_nodes")


def robust_decision(features, thresholds, noise_frac=0.10, depth_limit=4, limit=4):
    """Run the game with BOTH solvers and report the inspector's decision.

    decision = the first move of optimal play: TRUST, REVIEW or
    REMEASURE(f) (re-read f, then trust). Uses only the measured features.
    """
    facts = thresholds.discretise(features)
    fired = forward_chain(facts).fired
    border = borderline_features(features, thresholds, fired, noise_frac, limit)
    game = RobustnessGame(features, thresholds, border, depth_limit)
    mm_nodes, ab_nodes = [0], [0]
    mm_value, mm_move = minimax(game, game.initial(), mm_nodes)
    ab_value, ab_move = alphabeta(game, game.initial(), ab_nodes)
    assert (mm_value, mm_move) == (ab_value, ab_move), "alpha-beta disagrees with minimax"
    name = game.move_name(ab_move)
    return RobustResult(name.split("(")[0], ab_value / 10.0, name, game.nominal,
                        [n for n, _ in border], mm_nodes[0], ab_nodes[0])
