"""Standard Module 1 puzzles, used to sanity-check the generic search engine."""
import random

from search import Problem

GOAL_8 = (1, 2, 3, 4, 5, 6, 7, 8, 0)          # 0 is the blank


class EightPuzzle(Problem):
    """State: tuple of 9 numbers (row-major). Action: slide the blank."""

    MOVES = {"U": -3, "D": 3, "L": -1, "R": 1}

    def __init__(self, start): self.start = tuple(start)
    def initial_state(self): return self.start
    def goal_test(self, state): return state == GOAL_8

    def actions(self, state):
        b = state.index(0)
        acts = []
        if b >= 3: acts.append("U")
        if b <= 5: acts.append("D")
        if b % 3 > 0: acts.append("L")
        if b % 3 < 2: acts.append("R")
        return acts

    def result(self, state, action):
        b = state.index(0)
        t = b + self.MOVES[action]
        s = list(state)
        s[b], s[t] = s[t], s[b]
        return tuple(s)


def manhattan(state):
    """Sum of tile distances to their goal squares (blank excluded).
    Admissible: each move shifts one tile by one square, so at least this
    many moves are needed. O(9)."""
    total = 0
    for pos, tile in enumerate(state):
        if tile:
            goal = tile - 1
            total += abs(pos // 3 - goal // 3) + abs(pos % 3 - goal % 3)
    return total


def scramble(moves, seed):
    """Start state reached by `moves` random legal moves from the goal
    (always solvable). Deterministic for a seed."""
    rng, problem, state = random.Random(seed), EightPuzzle(GOAL_8), GOAL_8
    for _ in range(moves):
        state = problem.result(state, rng.choice(problem.actions(state)))
    return state


class WaterJug(Problem):
    """Jugs of capacity (a, b); get exactly `target` litres in either jug.
    State (x, y). Actions: fill, empty, pour."""

    def __init__(self, cap_a=4, cap_b=3, target=2):
        self.cap, self.target = (cap_a, cap_b), target

    def initial_state(self): return (0, 0)
    def goal_test(self, s): return self.target in s

    def actions(self, s):
        return ["fill A", "fill B", "empty A", "empty B", "pour A>B", "pour B>A"]

    def result(self, s, action):
        x, y = s
        ca, cb = self.cap
        if action == "fill A": return (ca, y)
        if action == "fill B": return (x, cb)
        if action == "empty A": return (0, y)
        if action == "empty B": return (x, 0)
        if action == "pour A>B":
            d = min(x, cb - y)
            return (x - d, y + d)
        d = min(y, ca - x)
        return (x + d, y - d)


class MissionariesCannibals(Problem):
    """3 missionaries + 3 cannibals cross with a 2-seat boat; cannibals may
    never outnumber missionaries on a bank (unless there are no missionaries).
    State: (missionaries left, cannibals left, boat on left?)."""

    BOAT = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    def initial_state(self): return (3, 3, True)
    def goal_test(self, s): return s == (0, 0, False)

    @staticmethod
    def safe(m, c):
        return 0 <= m <= 3 and 0 <= c <= 3 and (m == 0 or m >= c) and (3 - m == 0 or 3 - m >= 3 - c)

    def actions(self, s):
        m, c, left = s
        sign = -1 if left else 1
        return [a for a in self.BOAT if self.safe(m + sign * a[0], c + sign * a[1])]

    def result(self, s, a):
        m, c, left = s
        sign = -1 if left else 1
        return (m + sign * a[0], c + sign * a[1], not left)
