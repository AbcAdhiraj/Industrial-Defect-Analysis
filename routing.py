"""Module 1 - the inspection-route problem (uses search.py).

A plate needs K checkpoints visited (the fault frame's required_checks plus the
mandatory ALIGNMENT checkpoint). Each checkpoint has 2-D coordinates drawn from
a seed. The inspection head starts at the DOCK at (0, 0).
  State  = (current checkpoint, frozenset of visited checkpoints)
  Action = move to an unvisited checkpoint whose PRECEDENCE predecessors are done
  Cost   = Euclidean distance        Goal = every checkpoint visited
Precedence: ALIGNMENT before everything else, plus the "A before B" pairs of
kb.PRECEDENCE_PAIRS (when both checks are present), enforced in `actions`.
"""
import itertools
import math
import random

from kb import MANDATORY_CHECK, PRECEDENCE_PAIRS
from search import Problem

DOCK = "DOCK"


def make_checkpoints(required_checks, seed):
    """{name: (x, y)} on a 100x100 plate for ALIGNMENT + required checks, plus
    the DOCK at the origin. Deterministic in `seed`. O(K)."""
    rng = random.Random(seed)
    names = [MANDATORY_CHECK] + [c for c in required_checks if c != MANDATORY_CHECK]
    points = {n: (round(rng.uniform(0, 100), 1), round(rng.uniform(0, 100), 1)) for n in names}
    points[DOCK] = (0.0, 0.0)
    return points


def precedence_for(names):
    """Set of (before, after) pairs among `names`."""
    pairs = {(MANDATORY_CHECK, n) for n in names if n != MANDATORY_CHECK}
    pairs |= {(a, b) for a, b in PRECEDENCE_PAIRS if a in names and b in names}
    return pairs


def dist(points, a, b):
    (x1, y1), (x2, y2) = points[a], points[b]
    return math.hypot(x1 - x2, y1 - y2)


class InspectionRoute(Problem):
    def __init__(self, points, precedence=None):
        self.points = points
        self.names = sorted(n for n in points if n != DOCK)
        self.precedence = precedence if precedence is not None else precedence_for(self.names)
        self.before = {n: {a for a, b in self.precedence if b == n} for n in self.names}

    def initial_state(self): return (DOCK, frozenset())
    def goal_test(self, state): return len(state[1]) == len(self.names)

    def actions(self, state):
        """Unvisited checkpoints whose predecessors are all visited. O(K)."""
        visited = state[1]
        return [n for n in self.names if n not in visited and self.before[n] <= visited]

    def result(self, state, action): return (action, state[1] | {action})
    def step_cost(self, state, action, next_state): return dist(self.points, state[0], action)


def _mst_cost(points, nodes):
    """Prim's minimum spanning tree weight over `nodes`. O(m^2)."""
    nodes = list(nodes)
    if len(nodes) < 2:
        return 0.0
    best = {n: dist(points, nodes[0], n) for n in nodes[1:]}
    total = 0.0
    while best:
        n = min(best, key=best.get)
        total += best.pop(n)
        for m in best:
            best[m] = min(best[m], dist(points, n, m))
    return total


def make_heuristics(problem):
    """Return (h_strong, h_weak) for `problem`.

    h_strong(s) = MST(unvisited) + distance(current, nearest unvisited).
    ADMISSIBLE: any finishing route walks current -> u1 -> u2 -> ... -> um over
    all unvisited u's. The edges u1-u2, ..., u(m-1)-um connect all unvisited
    checkpoints, i.e. they contain a spanning tree of them, so their length
    is >= MST(unvisited). The first leg current->u1 is >= the distance to the
    nearest unvisited checkpoint. Hence true remaining cost >= h_strong. The
    precedence rules only remove routes, which can only RAISE the true cost,
    so the bound still holds. (h = 0 at the goal.)
    h_weak(s) = distance(current, nearest unvisited): also admissible (it is
    just the first term) but much less informed.
    """
    pts, names = problem.points, problem.names

    def unvisited(state): return [n for n in names if n not in state[1]]

    def h_weak(state):
        rest = unvisited(state)
        return min(dist(pts, state[0], n) for n in rest) if rest else 0.0

    def h_strong(state):
        rest = unvisited(state)
        return h_weak(state) + _mst_cost(pts, rest) if rest else 0.0

    return h_strong, h_weak


def brute_force_optimal(problem):
    """Cheapest route by trying every permutation that respects precedence.
    Used ONLY to verify the search algorithms, for K <= 8. O(K! * K)."""
    if len(problem.names) > 8:
        raise ValueError("brute force limited to K <= 8")
    best = float("inf")
    for order in itertools.permutations(problem.names):
        pos = {n: i for i, n in enumerate(order)}
        if any(pos[a] > pos[b] for a, b in problem.precedence):
            continue
        cost, here = 0.0, DOCK
        for n in order:
            cost += dist(problem.points, here, n)
            here = n
        best = min(best, cost)
    return best


def random_problem(k, seed):
    """Random plate with K checkpoints (ALIGNMENT + K-1 other checks) for tests
    and benchmarks. The checks are sampled from the catalogue."""
    from kb import CHECKS
    rng = random.Random(seed)
    others = sorted(c for c in CHECKS if c != MANDATORY_CHECK)
    chosen = rng.sample(others, k - 1)
    return InspectionRoute(make_checkpoints(chosen, seed * 7919 + k))
