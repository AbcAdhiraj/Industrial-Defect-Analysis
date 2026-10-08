"""Plain-Python test runner (no framework): python3 tests/test_all.py
Prints PASS/FAIL per group; exit code 1 if any check fails."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import classic_problems as cp
import csp as csp_mod
import data_loader as dl
import evaluation
import inference as inf
import kb
import robust_check as rc
import routing
import scheduling as sc
import search as se
from pipeline import Pipeline
from rules import RULES

DATA = dl.load_dataset(download=False)
TRAIN, TEST = dl.stratified_split(DATA.plates, seed=42)
TH = kb.Thresholds(TRAIN)
GROUPS, FAILED = [], []


def group(fn):
    GROUPS.append(fn)
    return fn


class Checker:
    def __init__(self): self.n, self.bad = 0, []

    def __call__(self, condition, message):
        self.n += 1
        if not condition:
            self.bad.append(message)


@group
def data_loader_and_split(c):
    matrix = dl.make_synthetic_matrix(seed=1)
    counts = dl.validate_matrix(matrix)
    c(dict(counts) == dl.SYNTH_COUNTS, "validated class counts")
    c(dl.make_synthetic_matrix(seed=1) == matrix, "synthetic generator is deterministic")
    bad = [list(r) for r in matrix]
    bad[5][27:] = [1.0] * 7
    try:
        dl.validate_matrix(bad)
        c(False, "validate_matrix must reject a row with several class flags")
    except ValueError:
        c(True, "")
    c(len(DATA.plates) == 1941 and len(TRAIN) + len(TEST) == 1941, "all 1941 plates split")
    expected_train = sum(round(n * 0.7) for n in dl.class_counts(DATA.plates).values())
    c(len(TRAIN) == expected_train, "train size is 70% per class")
    c({p.label for p in TRAIN} == set(dl.CLASSES) == {p.label for p in TEST}, "every class in both splits")
    c(not {p.pid for p in TRAIN} & {p.pid for p in TEST}, "train/test disjoint")
    again, _ = dl.stratified_split(DATA.plates, seed=42)
    c([p.pid for p in again] == [p.pid for p in TRAIN], "split deterministic for a seed")


@group
def discretisation(c):
    c(kb.quantile([1, 2, 3, 4, 5], 0.25) == 2.0 and kb.quantile([1, 2, 3, 4, 5], 0.5) == 3.0, "quantile")
    c(abs(kb.quantile([0, 10], 0.25) - 2.5) < 1e-12, "quantile interpolates")
    low, high, _ = TH.table["X_Perimeter"]
    c(TH.level("X_Perimeter", low) == "LOW" and TH.level("X_Perimeter", high) == "MEDIUM"
      and TH.level("X_Perimeter", high + 1e-6) == "HIGH", "level boundaries")
    c("TypeOfSteel_A300" in TH.binary and TH.level("TypeOfSteel_A300", 1.0) == "HIGH"
      and TH.level("TypeOfSteel_A300", 0.0) == "LOW", "binary flags map 0->LOW, 1->HIGH")
    th2 = kb.Thresholds(TRAIN)          # thresholds depend only on the train plates given
    c(th2.table == TH.table, "thresholds reproducible from train split")
    c(set(TH.discretise(TRAIN[0].features).values()) <= set(kb.LEVELS), "labels are LOW/MEDIUM/HIGH")


@group
def rule_base_shape(c):
    c(20 <= len(RULES) <= 35, "20-35 rules (found %d)" % len(RULES))
    c(len({r.rid for r in RULES}) == len(RULES), "rule ids unique")
    c(all(r.rid and r.text and r.why and isinstance(r.priority, int) for r in RULES), "rule fields filled")
    c(any(r.conclusion == ("fault", "Other_Faults") and not r.conditions and r.priority == 0 for r in RULES),
      "fallback rule exists")
    c({r.conclusion[1] for r in RULES if r.conclusion[0] == "fault"} == set(dl.CLASSES), "every fault concluded")


@group
def forward_chaining_hand_built(c):
    z = {"Y_Perimeter": "LOW", "X_Perimeter": "HIGH", "Steel_Plate_Thickness": "LOW"}
    res = inf.forward_chain(z)
    c(res.label == "Z_Scratch" and res.winner.rid == "F7", "Z_Scratch via F7 (priority 3)")
    c({"D1", "D6", "F7", "F10", "F21"} <= {r.rid for r in res.fired}, "all applicable rules fired")
    c(res.facts["ELONGATED_X"] == "YES", "symptom derived (chained)")
    c(inf.forward_chain({}).label == "Other_Faults", "fallback when nothing applies")
    stain = {"Steel_Plate_Thickness": "HIGH", "Outside_X_Index": "HIGH", "TypeOfSteel_A400": "HIGH"}
    c(inf.forward_chain(stain).winner.rid == "F1", "3-condition priority-3 rule wins")
    # specificity tie-break with custom rules of equal priority
    a = kb.Rule("A", "", 1, (("x", frozenset({"1"})),), ("fault", "Pastry"), "")
    b = kb.Rule("B", "", 1, (("x", frozenset({"1"})), ("y", frozenset({"1"}))), ("fault", "Bumps"), "")
    c(inf.forward_chain({"x": "1", "y": "1"}, [a, b]).label == "Bumps", "specificity breaks priority tie")
    c(inf.forward_chain({"x": "1", "y": "1"}, [a, b]).conflicts == [a], "conflict set reported")
    text = inf.explain(TEST[0].features, TH)
    c("Rules fired" in text and "Decision:" in text and "Proof" in text, "explain() has all sections")


@group
def backward_chaining_proof(c):
    z = {"Y_Perimeter": "LOW", "X_Perimeter": "HIGH", "Steel_Plate_Thickness": "LOW"}
    proof = inf.backward_chain(("fault", "Z_Scratch"), z)
    c(proof is not None and proof.rule.rid == "F7", "proof uses highest-priority supporting rule")
    c(proof.children[0].rule.rid == "D1", "symptom proved by its own rule (depth 2)")
    c(inf.backward_chain(("fault", "K_Scatch"), z) is None, "unprovable goal -> None")
    c(inf.backward_chain(("fault", "Other_Faults"), {}) is not None, "fallback provable from no facts")
    needed = dict(inf.facts_needed(("fault", "Z_Scratch")))
    c(("Steel_Plate_Thickness", ["LOW"]) in needed["F7"], "facts_needed expands symptoms to base facts")
    c("fault = Z_Scratch" in inf.format_proof(proof), "proof printable")
    loop = [kb.Rule("L1", "", 1, (("P", frozenset({"YES"})),), ("Q", "YES"), ""),
            kb.Rule("L2", "", 1, (("Q", frozenset({"YES"})),), ("P", "YES"), "")]
    c(inf.backward_chain(("P", "YES"), {}, loop) is None, "cyclic rules do not loop forever")


@group
def semantic_net_and_frames(c):
    net = kb.build_semantic_net()
    c(net.is_a("Z_Scratch", "SurfaceDefect") and net.is_a("Z_Scratch", "Fault"), "Z_Scratch is-a SurfaceDefect is-a Fault")
    c(net.ancestors("Bumps") == ["ShapeDefect", "Fault"], "ancestors walk is-a links")
    c(not net.is_a("Bumps", "SurfaceDefect"), "negative query")
    c(net.members("SurfaceDefect") == ["K_Scatch", "Z_Scratch"], "members of a category")
    c(net.inherited("Pastry", "detected-by") == "LASER" and net.inherited("Other_Faults", "detected-by") == "CAMERA",
      "property inheritance")
    frames = kb.build_fault_frames()
    c(set(frames) == set(dl.CLASSES), "one FaultType frame per class")
    c(all(f.get("severity") and f.get("typical_conditions") and f.get("required_checks") for f in frames.values()),
      "FaultType slots filled")
    c(kb.plate_frame(7, TH.discretise(TEST[0].features)).get("X_Perimeter") in kb.LEVELS, "Plate frame slots")
    c(kb.Frame("child", frames["Bumps"]).get("severity") == frames["Bumps"].get("severity"), "frame inheritance")


@group
def minimax_equals_alphabeta_random_trees(c):
    rng = random.Random(12345)
    saved = 0
    for i in range(200):
        game = rc.TreeGame(rc.random_tree(rng, rng.randint(1, 6), 4))
        a, b = [0], [0]
        v1 = rc.minimax(game, game.initial(), a)
        v2 = rc.alphabeta(game, game.initial(), b)
        c(v1 == v2, "tree %d: minimax %s vs alpha-beta %s" % (i, v1, v2))
        c(b[0] <= a[0], "tree %d: alpha-beta expanded more nodes" % i)
        saved += a[0] - b[0]
    c(saved > 0, "alpha-beta saves nodes overall")


@group
def minimax_equals_alphabeta_real_plates(c):
    decisions = set()
    for p in TEST[:150]:
        game = rc.make_game(p.features, TH)
        a, b = [0], [0]
        v1 = rc.minimax(game, game.initial(), a)
        v2 = rc.alphabeta(game, game.initial(), b)
        c(v1 == v2, "plate %d: value/move differ %s vs %s" % (p.pid, v1, v2))
        c(b[0] <= a[0], "plate %d: alpha-beta more nodes" % p.pid)
        decisions.add(v2[1] if isinstance(v2[1], str) else v2[1][0])
    c({"TRUST", "REVIEW"} <= decisions, "game produces both TRUST and REVIEW (got %s)" % sorted(decisions))
    flips = 0
    for p in TEST[:150]:
        r = rc.robust_decision(p.features, TH)
        if r.decision == "REVIEW":                      # a +/-delta change really can flip the label
            game = rc.make_game(p.features, TH)
            n = len(game.names)
            labels = set()
            for signs in itertools.product((-1, 0, 1), repeat=n):
                s = (0, signs, (False,) * n, 0, True, False)
                labels.add(game._label(s))
            flips += len(labels) > 1
    c(flips > 0, "REVIEW plates can really change label under noise")
    plain = [p for p in TEST[:150] if not rc.make_game(p.features, TH).names]
    c(all(rc.robust_decision(p.features, TH).decision == "TRUST" for p in plain), "no borderline feature -> TRUST")


@group
def generic_search_8_puzzle_and_classics(c):
    for seed in range(5):
        puzzle = cp.EightPuzzle(cp.scramble(14, seed))
        bfs, astar = se.breadth_first(puzzle), se.a_star(puzzle, cp.manhattan)
        ids = se.iterative_deepening(puzzle, 20)
        c(astar.cost <= bfs.cost, "A* cost <= BFS cost (seed %d)" % seed)
        c(len(bfs.path) == len(astar.path) == len(ids.path), "all optimal in moves (seed %d)" % seed)
        c(astar.nodes_expanded <= bfs.nodes_expanded, "A* expands no more than BFS (seed %d)" % seed)
        state = puzzle.start
        for a in astar.path:
            state = puzzle.result(state, a)
        c(puzzle.goal_test(state), "A* path reaches goal")
    c(len(se.breadth_first(cp.WaterJug(4, 3, 2)).path) == 4, "water jug 4/3 -> 2 in 4 moves")
    c(se.breadth_first(cp.MissionariesCannibals()).cost == 11, "missionaries & cannibals: 11 crossings")
    capped = se.breadth_first(cp.EightPuzzle(cp.scramble(30, 1)), se.Limits(100, None, None))
    c(capped.status.startswith("capped"), "node cap stops search with a clear status")


def _routes(ks=(4, 5, 6, 7, 8), per_k=8):
    for k in ks:
        for i in range(per_k):
            yield k, routing.random_problem(k, 500 + 31 * k + i)


@group
def routing_astar_ucs_equal_brute_force(c):
    for k, problem in _routes():
        best = routing.brute_force_optimal(problem)
        hs, hw = routing.make_heuristics(problem)
        for name, res in (("UCS", se.uniform_cost(problem)), ("A*", se.a_star(problem, hs)),
                          ("A*weak", se.a_star(problem, hw))):
            c(res.status == "solved" and abs(res.cost - best) < 1e-6, "K=%d %s cost %.3f vs brute %.3f" % (k, name, res.cost, best))
        pos = {n: i for i, n in enumerate(se.a_star(problem, hs).path)}
        c(all(pos[a] < pos[b] for a, b in problem.precedence), "K=%d precedence respected" % k)
        c(problem.names[0] == "ALIGNMENT" and se.a_star(problem, hs).path[0] == "ALIGNMENT", "alignment visited first")
        c(se.breadth_first(problem).cost >= best - 1e-9, "BFS cost never below optimum")


@group
def mst_heuristic_admissible(c):
    for k in (4, 5, 6, 7):
        for i in range(10):
            problem = routing.random_problem(k, 900 + 17 * k + i)
            hs, hw = routing.make_heuristics(problem)
            memo = {}

            def to_go(state):
                if problem.goal_test(state):
                    return 0.0
                if state not in memo:
                    memo[state] = min(problem.step_cost(state, a, problem.result(state, a))
                                      + to_go(problem.result(state, a)) for a in problem.actions(state))
                return memo[state]
            to_go(problem.initial_state())
            for state, truth in memo.items():
                c(hs(state) <= truth + 1e-9, "K=%d MST h=%.2f > true %.2f" % (k, hs(state), truth))
                c(hw(state) <= hs(state) + 1e-9, "weak <= strong")
    c(True, "")


@group
def csp_solvers_agree(c):
    known = {2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}
    for n, expected in known.items():
        for m in ("bt", "fc", "fc_mrv", "ac3_fc_mrv"):
            if m == "bt" and n > 8:
                continue
            res = csp_mod.solve(csp_mod.n_queens(n), m, all_solutions=True)
            c(len(res.solutions) == expected, "%d-queens %s: %d solutions, expected %d" % (n, m, len(res.solutions), expected))
            c((res.status == "solved") == (expected > 0), "%d-queens %s solvability" % (n, m))
    for sol in csp_mod.solve(csp_mod.n_queens(6), "fc_mrv", all_solutions=True).solutions:
        c(all(sol[i] != sol[j] and abs(sol[i] - sol[j]) != j - i for i in range(6) for j in range(i + 1, 6)), "valid queens")
    regions = sorted(csp_mod.AUSTRALIA)
    brute = sum(all(a[regions.index(r)] != a[regions.index(o)] for r, ns in csp_mod.AUSTRALIA.items() for o in ns)
                for a in itertools.product("RGB", repeat=len(regions)))
    for m in ("bt", "fc", "fc_mrv", "ac3_fc_mrv"):
        c(len(csp_mod.solve(csp_mod.map_coloring(csp_mod.AUSTRALIA, list("RGB")), m, True).solutions) == brute, "map colouring %s" % m)
        c(csp_mod.solve(csp_mod.map_coloring(csp_mod.AUSTRALIA, list("RG")), m).status == "unsatisfiable", "2 colours impossible")
    fc, mrv = csp_mod.solve(csp_mod.n_queens(8), "fc", True), csp_mod.solve(csp_mod.n_queens(8), "bt", True)
    c(fc.nodes < mrv.nodes, "forward checking visits fewer nodes than plain backtracking")


@group
def scheduling_constraints_hold(c):
    solved = unsat = 0
    for n in (2, 3, 4, 5):
        for seed in range(4):
            inst = sc.make_instance(n, 3, 2 * n + 4, seed)
            results = {m: csp_mod.solve(sc.build_csp(inst), m, node_limit=30000) for m in ("fc_mrv", "ac3_fc_mrv")}
            c(results["fc_mrv"].status == results["ac3_fc_mrv"].status or "limit" in
              (results["fc_mrv"].status, results["ac3_fc_mrv"].status), "solvers agree n=%d seed=%d" % (n, seed))
            for m, res in results.items():
                if res.solutions:
                    solved += 1
                    c(sc.verify_schedule(inst, res.solutions[0]) == [], "n=%d seed=%d %s violates a constraint" % (n, seed, m))
                    c(len(res.solutions[0]) == len(sc.build_csp(inst).variables), "every check scheduled")
    c(solved > 0, "some instances solved")
    inst = sc.make_instance(3, 3, 8, 0)
    sol = dict(csp_mod.solve(sc.build_csp(inst), "fc_mrv").solutions[0])
    first, second = list(sol)[:2]
    sol[second] = sol[first]                                     # force a clash
    c(sc.verify_schedule(inst, sol) != [], "verifier detects a corrupted schedule")
    tiny = sc.make_instance(3, 3, 2, 0)                          # 3 ALIGNMENTs, 2 slots, one laser
    for m in ("bt", "fc", "fc_mrv", "ac3_fc_mrv"):
        c(csp_mod.solve(sc.build_csp(tiny), m).status == "unsatisfiable", "impossible batch reported unsatisfiable (%s)" % m)


@group
def evaluation_and_pipeline(c):
    result = evaluation.evaluate_all(TRAIN, TEST, TH)
    c(sum(map(sum, result["confusion"])) == len(TEST), "confusion matrix counts every test plate")
    c(abs(sum(result["confusion"][i][i] for i in range(7)) / len(TEST) - result["expert_accuracy"]) < 1e-12, "diagonal = accuracy")
    c(result["majority_class"] == max(dl.CLASSES, key=lambda k: sum(p.label == k for p in TRAIN)), "majority class from TRAIN")
    c(all(0 <= result[k] <= 1 for k in ("expert_accuracy", "majority_accuracy", "single_accuracy")), "accuracies in [0,1]")
    pipe = Pipeline(DATA)
    reports = [pipe.run_plate(p) for p in pipe.sample(5, seed=3)]
    for r in reports:
        c(sorted(r.route) == sorted(r.checks) and r.route[0] == "ALIGNMENT", "route visits every required check, alignment first")
        c(set(pipe.frames[r.label].get("required_checks")) <= set(r.checks), "frame required_checks feed the route")
    inst, res = pipe.schedule(reports)
    c(res.solutions and sc.verify_schedule(inst, res.solutions[0]) == [], "batch schedule valid")


def main():
    print("Running tests on %s\n" % dl.source_tag(DATA))
    for fn in GROUPS:
        checker = Checker()
        try:
            fn(checker)
        except Exception as exc:                                  # a crash is a failure
            checker.bad.append("exception: %r" % exc)
        status = "PASS" if not checker.bad else "FAIL"
        print("[%s] %-48s %d checks" % (status, fn.__name__, checker.n))
        for msg in checker.bad[:5]:
            print("      -", msg)
        if checker.bad:
            FAILED.append(fn.__name__)
    print("\n%d groups, %d passed, %d failed" % (len(GROUPS), len(GROUPS) - len(FAILED), len(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
