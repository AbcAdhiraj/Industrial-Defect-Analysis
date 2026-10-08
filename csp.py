"""Module 1 - constraint satisfaction: a generic binary CSP and four solvers.

CSP = variables + a finite domain per variable + binary constraints (a
predicate over the values of two variables). A solution assigns every
variable a value so that all constraints hold.

Solvers (selected by name):
  "bt"          plain backtracking, fixed variable order
  "fc"          backtracking + forward checking, fixed variable order
  "fc_mrv"      forward checking + minimum-remaining-values ordering
  "ac3_fc_mrv"  AC-3 arc-consistency preprocessing, then fc_mrv

Counters: nodes = value assignments actually made (after the consistency
check / forward check passes); backtracks = variable levels that failed after
trying all their values; checks = constraint-predicate evaluations.
"""
import time
from collections import deque, namedtuple

CSPResult = namedtuple("CSPResult", "status solutions nodes backtracks checks seconds")


class LimitReached(Exception):
    pass


class CSP:
    def __init__(self, variables, domains):
        self.variables = list(variables)
        self.domains = {v: list(domains[v]) for v in self.variables}
        self.constraints = {}                       # (x, y) -> predicate(vx, vy)
        self.neighbors = {v: [] for v in self.variables}

    def add_constraint(self, x, y, predicate):
        """Register predicate(value_x, value_y) in both directions."""
        self.constraints[(x, y)] = predicate
        self.constraints[(y, x)] = lambda b, a, p=predicate: p(a, b)
        for u, w in ((x, y), (y, x)):
            if w not in self.neighbors[u]:
                self.neighbors[u].append(w)

    def ok(self, x, vx, y, vy):
        predicate = self.constraints.get((x, y))
        return predicate is None or predicate(vx, vy)


def revise(csp, domains, x, y, counters):
    """Delete values of x that have no supporting value in y. O(d^2).
    Returns True if the domain of x shrank."""
    kept = []
    for vx in domains[x]:
        support = False
        for vy in domains[y]:
            counters["checks"] += 1
            if csp.ok(x, vx, y, vy):
                support = True
                break
        if support:
            kept.append(vx)
    changed = len(kept) < len(domains[x])
    domains[x] = kept
    return changed


def ac3(csp, domains, counters):
    """AC-3 arc consistency. Returns False if some domain becomes empty.

    Purpose: prune values that can never be part of a solution BEFORE search.
    Time O(e * d^3) worst case (e arcs, d domain size), space O(e).
    WHY it is sound: a value with no support in a neighbour's domain cannot
    appear in any solution, so deleting it loses nothing; when x's domain
    shrinks, arcs (z, x) are re-queued because z's supports may have vanished.
    """
    queue = deque(csp.constraints)
    while queue:
        x, y = queue.popleft()
        if revise(csp, domains, x, y, counters):
            if not domains[x]:
                return False
            queue.extend((z, x) for z in csp.neighbors[x] if z != y)
    return True


def solve(csp, method="fc_mrv", all_solutions=False, node_limit=None):
    """Backtracking search with the chosen strategy. Returns CSPResult.

    Purpose: find one (or all) solutions. Worst-case time O(d^n) for n
    variables and domain size d, space O(n*d) (recursion + saved domains);
    the strategies only change how much of that tree is visited.
    WHY the improvements help: forward checking deletes, right after each
    assignment, the values it rules out in unassigned neighbours, so doomed
    branches die one level earlier; MRV picks the variable with the fewest
    legal values ("fail first"), which exposes dead ends near the root where
    pruning is cheapest; AC-3 shrinks domains before the first assignment.
    """
    start = time.perf_counter()
    counters = {"nodes": 0, "backtracks": 0, "checks": 0}
    domains = {v: list(d) for v, d in csp.domains.items()}
    use_fc = method != "bt"
    use_mrv = method in ("fc_mrv", "ac3_fc_mrv")
    solutions, assignment = [], {}
    order = {v: i for i, v in enumerate(csp.variables)}

    def finish(status):
        return CSPResult(status, solutions, counters["nodes"], counters["backtracks"],
                         counters["checks"], time.perf_counter() - start)

    if method == "ac3_fc_mrv" and not ac3(csp, domains, counters):
        return finish("unsatisfiable")

    def pick():
        free = [v for v in csp.variables if v not in assignment]
        if not use_mrv:
            return free[0]
        return min(free, key=lambda v: (len(domains[v]),
                                        -sum(w not in assignment for w in csp.neighbors[v]),
                                        order[v]))

    def forward_check(var, val):
        """Prune neighbours' domains; return undo list or None on wipe-out."""
        undo = []
        for y in csp.neighbors[var]:
            if y in assignment:
                continue
            new = []
            for w in domains[y]:
                counters["checks"] += 1
                if csp.ok(var, val, y, w):
                    new.append(w)
            if len(new) < len(domains[y]):
                undo.append((y, domains[y]))
                domains[y] = new
                if not new:
                    return undo, False
        return undo, True

    def search():
        if len(assignment) == len(csp.variables):
            solutions.append(dict(assignment))
            return not all_solutions
        var = pick()
        before = len(solutions)
        for val in list(domains[var]):
            if not use_fc:                           # plain BT checks against assigned vars
                ok = True
                for y in csp.neighbors[var]:
                    if y in assignment:
                        counters["checks"] += 1
                        if not csp.ok(var, val, y, assignment[y]):
                            ok = False
                            break
                if not ok:
                    continue
            undo, alive = (None, True) if not use_fc else forward_check(var, val)
            if alive:
                counters["nodes"] += 1
                if node_limit and counters["nodes"] > node_limit:
                    raise LimitReached()
                assignment[var] = val
                if search():
                    return True
                del assignment[var]
            for y, old in reversed(undo or []):
                domains[y] = old
        if len(solutions) == before:                 # level produced nothing: a dead end
            counters["backtracks"] += 1
        return False

    try:
        search()
    except LimitReached:
        return finish("limit")
    return finish("solved" if solutions else "unsatisfiable")


# ----------------------------------------------------- standard test problems
def n_queens(n):
    """Variable i = row of the queen in column i; domain = rows 0..n-1."""
    csp = CSP(range(n), {i: range(n) for i in range(n)})
    for i in range(n):
        for j in range(i + 1, n):
            csp.add_constraint(i, j, lambda a, b, d=j - i: a != b and abs(a - b) != d)
    return csp


AUSTRALIA = {"WA": ["NT", "SA"], "NT": ["WA", "SA", "Q"], "SA": ["WA", "NT", "Q", "NSW", "V"],
             "Q": ["NT", "SA", "NSW"], "NSW": ["Q", "SA", "V"], "V": ["SA", "NSW"], "T": []}


def map_coloring(adjacency, colors):
    """Adjacent regions must get different colours."""
    csp = CSP(sorted(adjacency), {r: colors for r in adjacency})
    for region, nbrs in adjacency.items():
        for other in nbrs:
            if region < other:
                csp.add_constraint(region, other, lambda a, b: a != b)
    return csp
