"""Module 1 - generic state-space search framework.

A problem is described by five methods (the textbook formulation):
initial_state, actions, result, goal_test, step_cost. The algorithms below
know nothing about the problem, so the same code solves the 8-puzzle, the
water-jug puzzle and the inspection-route problem.

"Nodes expanded" counts nodes popped from the frontier and expanded (the goal
node, if popped, is counted). BFS tests the goal when a child is generated
(textbook), the others when a node is popped (needed for cost-optimality).
"""
import heapq
import time
from collections import deque, namedtuple


class Problem:
    """Interface. States must be hashable; actions(s) must be deterministic."""

    def initial_state(self): raise NotImplementedError
    def actions(self, state): raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def goal_test(self, state): raise NotImplementedError
    def step_cost(self, state, action, next_state): return 1


SearchResult = namedtuple("SearchResult", "status path cost nodes_expanded max_frontier seconds")
Limits = namedtuple("Limits", "max_nodes max_frontier max_seconds")
NO_LIMITS = Limits(None, None, None)


class CapExceeded(Exception):
    """Raised when a node, frontier (memory) or time cap is hit."""


class Node:
    __slots__ = ("state", "parent", "action", "g", "depth")

    def __init__(self, state, parent=None, action=None, g=0.0):
        self.state, self.parent, self.action, self.g = state, parent, action, g
        self.depth = 0 if parent is None else parent.depth + 1


class _Run:
    """Bookkeeping shared by all algorithms: counters, clock and caps."""

    def __init__(self, limits):
        self.limits, self.nodes, self.max_frontier = limits or NO_LIMITS, 0, 0
        self.start = time.perf_counter()

    def expand(self, frontier_size):
        self.nodes += 1
        self.max_frontier = max(self.max_frontier, frontier_size)
        lim = self.limits
        if lim.max_nodes and self.nodes > lim.max_nodes:
            raise CapExceeded("node cap")
        if lim.max_frontier and frontier_size > lim.max_frontier:
            raise CapExceeded("memory cap")
        if lim.max_seconds and self.nodes % 256 == 0 and \
                time.perf_counter() - self.start > lim.max_seconds:
            raise CapExceeded("time cap")

    def finish(self, status, node=None):
        path, cost = [], float("nan")
        if node is not None:
            cost, walker = node.g, node
            while walker.parent is not None:
                path.append(walker.action)
                walker = walker.parent
            path.reverse()
        return SearchResult(status, path, cost, self.nodes, self.max_frontier,
                            time.perf_counter() - self.start)


def _child(problem, node, action):
    nxt = problem.result(node.state, action)
    return Node(nxt, node, action, node.g + problem.step_cost(node.state, action, nxt))


def breadth_first(problem, limits=None):
    """BFS graph search. Finds the solution with the FEWEST steps.

    Time/space O(b^d) in the worst case (b = branching factor, d = solution
    depth); the visited set stores every state seen. WHY it is step-optimal:
    the FIFO queue expands all depth-k nodes before any depth-(k+1) node, so
    the first goal generated has minimum depth. It is NOT cost-optimal when
    step costs differ.
    """
    run = _Run(limits)
    root = Node(problem.initial_state())
    if problem.goal_test(root.state):
        return run.finish("solved", root)
    frontier, reached = deque([root]), {root.state}
    try:
        while frontier:
            node = frontier.popleft()
            run.expand(len(frontier) + 1)
            for action in problem.actions(node.state):
                child = _child(problem, node, action)
                if child.state in reached:
                    continue
                if problem.goal_test(child.state):
                    return run.finish("solved", child)
                reached.add(child.state)
                frontier.append(child)
    except CapExceeded as cap:
        return run.finish("capped: %s" % cap)
    return run.finish("no solution")


def _depth_limited(problem, limit, run):
    """One depth-limited DFS pass. Returns (node or None, cutoff_happened).
    Cycle check: a state already on the current path is skipped. A GLOBAL
    visited set would be wrong here: a state first reached deep (and cut off)
    could block a later, shallower route to it."""
    stack, cutoff = [Node(problem.initial_state())], False
    while stack:
        node = stack.pop()
        run.expand(len(stack) + 1)
        if problem.goal_test(node.state):
            return node, cutoff
        if node.depth >= limit:
            cutoff = True
            continue
        on_path, walker = set(), node
        while walker is not None:
            on_path.add(walker.state)
            walker = walker.parent
        for action in reversed(list(problem.actions(node.state))):
            child = _child(problem, node, action)
            if child.state not in on_path:
                stack.append(child)
    return None, cutoff


def depth_first(problem, depth_limit=50, limits=None):
    """Depth-limited DFS (tree search with path-cycle check).

    Time O(b^l), space O(b*l) for limit l - the stack holds only one path
    plus its siblings. WHY it can be fast but poor: it dives to the first
    complete solution it meets, which is neither shortest nor cheapest.
    """
    run = _Run(limits)
    try:
        node, _ = _depth_limited(problem, depth_limit, run)
    except CapExceeded as cap:
        return run.finish("capped: %s" % cap)
    return run.finish("solved", node) if node else run.finish("no solution (within limit)")


def iterative_deepening(problem, max_depth=50, limits=None):
    """IDDFS: depth-limited DFS with limit 0, 1, 2, ...

    Time O(b^d), space O(b*d). WHY the repeated work is cheap: in a tree most
    nodes sit on the deepest level, so re-expanding the shallow levels adds
    only a constant factor, while memory stays linear like DFS and the first
    solution found has minimum depth like BFS (step-optimal, not cost-optimal).
    """
    run = _Run(limits)
    try:
        for limit in range(max_depth + 1):
            node, cutoff = _depth_limited(problem, limit, run)
            if node:
                return run.finish("solved", node)
            if not cutoff:
                return run.finish("no solution")
    except CapExceeded as cap:
        return run.finish("capped: %s" % cap)
    return run.finish("no solution (within limit)")


def best_first(problem, priority, reopen=True, limits=None):
    """Generic best-first graph search; `priority(g, state)` orders the frontier.

    Uniform-cost: priority = g.  Greedy: priority = h.  A*: priority = g + h.
    Time/space O(b^d) worst case (a good heuristic shrinks it a lot). With
    reopen=True a state is re-queued when a CHEAPER path to it is found, so
    UCS and A* (admissible h) return optimal-cost solutions. WHY: the
    frontier always holds the lowest-priority node; popping the goal with
    priority = its true cost means nothing cheaper remains unexplored.
    """
    run = _Run(limits)
    root = Node(problem.initial_state())
    best_g, counter = {root.state: 0.0}, 0
    heap = [(priority(0.0, root.state), counter, root)]
    try:
        while heap:
            _, _, node = heapq.heappop(heap)
            if reopen and node.g > best_g[node.state]:
                continue                                  # stale queue entry
            run.expand(len(heap) + 1)
            if problem.goal_test(node.state):
                return run.finish("solved", node)
            for action in problem.actions(node.state):
                child = _child(problem, node, action)
                known = best_g.get(child.state)
                if known is None or (reopen and child.g < known - 1e-12):
                    best_g[child.state] = child.g
                    counter += 1
                    heapq.heappush(heap, (priority(child.g, child.state), counter, child))
    except CapExceeded as cap:
        return run.finish("capped: %s" % cap)
    return run.finish("no solution")


def uniform_cost(problem, limits=None):
    """UCS = best-first with priority g (Dijkstra on the state graph)."""
    return best_first(problem, lambda g, s: g, True, limits)


def greedy_best_first(problem, heuristic, limits=None):
    """Greedy = best-first with priority h; fast but NOT optimal."""
    return best_first(problem, lambda g, s: heuristic(s), False, limits)


def a_star(problem, heuristic, limits=None):
    """A* = best-first with priority g + h; optimal if h never overestimates."""
    return best_first(problem, lambda g, s: g + heuristic(s), True, limits)
