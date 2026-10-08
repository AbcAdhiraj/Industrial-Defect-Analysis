"""All comparisons; writes CSVs to results/ and prints the tables.

"Mean of 5 runs": classifier and robustness results are deterministic on the
fixed split, so the 5 runs only average the TIMINGS. For routes and scheduling
each run r = 0..4 draws a different set of random instances (seed depends on
r); within a run every method receives IDENTICAL instances. A trailing '*'
marks a mean that includes runs stopped by a cap (then it is a lower bound).
Usage: python3 benchmarks.py [--quick]   (--quick = 1 run, 15 plates, for smoke tests)
"""
import csv
import os
import sys
import time

import csp as csp_mod
import evaluation
from data_loader import CLASSES, load_dataset, source_tag, stratified_split
from inference import classify
from kb import Thresholds
from robust_check import alphabeta, make_game, minimax
from routing import make_heuristics, random_problem
from scheduling import build_csp, make_instance
from search import (Limits, a_star, breadth_first, depth_first, greedy_best_first,
                    iterative_deepening, uniform_cost)

RUNS, PLATES_PER_K = 5, 60
SEARCH_LIMITS = Limits(200000, 500000, 5.0)      # nodes, frontier (memory), seconds
CSP_NODE_LIMIT = 20000
OUT = "results"
TAG = ["SYNTHETIC DATA"]                       # replaced in main() by the real source tag


def write_csv(name, header, rows):
    """Every CSV carries a data_source column (REAL DATA / SYNTHETIC DATA)."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header) + ["data_source"])
        writer.writerows(list(r) + [TAG[0]] for r in rows)


def show(title, header, rows, tag):
    print("\n=== %s [%s] ===" % (title, tag))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header] + rows:
        print("  ".join(str(x).rjust(w) for x, w in zip(line, widths)))


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def bench_classifier(train, test, th, tag):
    result = evaluation.evaluate_all(train, test, th)
    secs = []
    for _ in range(RUNS):
        t = time.perf_counter()
        [classify(p.features, th) for p in test]
        secs.append(time.perf_counter() - t)
    single = result["single_rule"]
    rows = [["expert system (30 rules)", "%.4f" % result["expert_accuracy"], "%.1f ms/run" % (1000 * mean(secs))],
            ["majority class (%s)" % result["majority_class"], "%.4f" % result["majority_accuracy"], "-"],
            ["best single feature (%s <= %.4g ? %s : %s)" % (single["feature"], single["threshold"],
                                                          single["left"], single["right"]),
             "%.4f" % result["single_accuracy"], "train acc %.4f" % single["train_accuracy"]]]
    show("1. Classifier accuracy on TEST (%d plates)" % len(test), ["method", "accuracy", "note"], rows, tag)
    print("\nConfusion matrix (rows = true, cols = predicted):\n" + evaluation.format_confusion(result["confusion"]))
    write_csv("classifier_accuracy.csv", ["method", "accuracy", "note"], rows)
    write_csv("confusion_matrix.csv", ["true\\pred"] + CLASSES,
              [[c] + r for c, r in zip(CLASSES, result["confusion"])])
    return result


def bench_robust(test, th, tag, plates=None):
    plates = plates or test
    truth = {p.pid: p.label for p in plates}
    mm_time, ab_time = [], []
    for _ in range(RUNS):
        for solver, bucket in ((minimax, mm_time), (alphabeta, ab_time)):
            t = time.perf_counter()
            for p in plates:
                game = make_game(p.features, th)
                solver(game, game.initial(), [0])
            bucket.append(time.perf_counter() - t)
    groups = {"TRUST": [], "REMEASURE": [], "REVIEW": []}
    mm_nodes = ab_nodes = 0
    for p in plates:
        game = make_game(p.features, th)
        a, b = [0], [0]
        v1, m1 = minimax(game, game.initial(), a)
        v2, m2 = alphabeta(game, game.initial(), b)
        assert (v1, m1) == (v2, m2)
        mm_nodes, ab_nodes = mm_nodes + a[0], ab_nodes + b[0]
        name = (m2 if isinstance(m2, str) else m2[0]).upper()
        groups[name].append(game.nominal == truth[p.pid])         # TRUE LABEL: evaluation only
    allhits = [h for g in groups.values() for h in g]
    acc = lambda g: "%.4f (n=%d)" % (mean(g), len(g)) if g else "n/a (n=0)"
    rows = [["accuracy, first move TRUST", acc(groups["TRUST"])],
            ["accuracy, TRUST + REMEASURE (proceed)", acc(groups["TRUST"] + groups["REMEASURE"])],
            ["accuracy, REVIEW plates", acc(groups["REVIEW"])],
            ["accuracy, all plates", acc(allhits)],
            ["REVIEW rate", "%.4f" % (len(groups["REVIEW"]) / len(plates))],
            ["REMEASURE rate", "%.4f" % (len(groups["REMEASURE"]) / len(plates))],
            ["minimax nodes (total)", mm_nodes], ["alpha-beta nodes (total)", ab_nodes],
            ["alpha-beta node saving", "%.1f%%" % (100 * (1 - ab_nodes / mm_nodes))],
            ["minimax time (mean of %d)" % RUNS, "%.3f s" % mean(mm_time)],
            ["alpha-beta time (mean of %d)" % RUNS, "%.3f s" % mean(ab_time)]]
    show("2. Robustness check on TEST (%d plates, noise=10%% of train std, depth 4)" % len(plates),
         ["metric", "value"], rows, tag)
    write_csv("robust_check.csv", ["metric", "value"], rows)


def bench_search(tag, runs, plates):
    methods = {"BFS": lambda p, hs, hw: breadth_first(p, SEARCH_LIMITS),
               "DFS": lambda p, hs, hw: depth_first(p, 50, SEARCH_LIMITS),
               "IDDFS": lambda p, hs, hw: iterative_deepening(p, 50, SEARCH_LIMITS),
               "UCS": lambda p, hs, hw: uniform_cost(p, SEARCH_LIMITS),
               "Greedy(MST)": lambda p, hs, hw: greedy_best_first(p, hs, SEARCH_LIMITS),
               "A*(MST)": lambda p, hs, hw: a_star(p, hs, SEARCH_LIMITS),
               "A*(weak)": lambda p, hs, hw: a_star(p, hw, SEARCH_LIMITS)}
    rows, skipped = [], set()
    for k in range(4, 10):
        stats = {m: {"nodes": [], "ms": [], "ratio": [], "opt": [], "cap": []} for m in methods}
        for run in range(runs):
            for i in range(plates):
                problem = random_problem(k, 100000 * run + 1000 * k + i)    # identical for all methods
                hs, hw = make_heuristics(problem)
                best = a_star(problem, hs).cost                              # reference optimum
                for name, fn in methods.items():
                    if name in skipped:
                        continue
                    r = fn(problem, hs, hw)
                    s = stats[name]
                    s["nodes"].append(r.nodes_expanded)
                    s["ms"].append(1000 * r.seconds)
                    s["cap"].append(r.status.startswith("capped"))
                    if r.status == "solved":
                        s["ratio"].append(r.cost / best)
                        s["opt"].append(r.cost <= best * (1 + 1e-9))
        for name in methods:
            s = stats[name]
            if name in skipped:
                rows.append([k, name, "SKIPPED (capped at smaller K)", "", "", "", ""])
                continue
            star = "*" if any(s["cap"]) else ""
            rows.append([k, name, "%.1f%s" % (mean(s["nodes"]), star), "%.3f" % mean(s["ms"]),
                         "%.3f" % mean(s["ratio"]), "%.0f%%" % (100 * mean(s["opt"])),
                         "%.0f%%" % (100 * mean(s["cap"]))])
            if mean(s["cap"]) >= 0.5:
                skipped.add(name)
    header = ["K", "method", "nodes", "ms", "cost/optimal", "optimal", "capped"]
    show("3. Inspection-route search (%d runs x %d plates per K; caps: %d nodes, %d frontier, %.0fs)"
         % (runs, plates, SEARCH_LIMITS.max_nodes, SEARCH_LIMITS.max_frontier, SEARCH_LIMITS.max_seconds),
         header, rows, tag)
    write_csv("search_routes.csv", header, rows)


def bench_csp(tag, runs):
    modes = ["bt", "fc", "fc_mrv", "ac3_fc_mrv"]
    header = ["plates", "checks", "method", "nodes", "backtracks", "ms", "sat/unsat/capped"]
    rows = []
    for n in range(3, 9):
        data = {m: {"nodes": [], "bt": [], "ms": [], "st": []} for m in modes}
        size = 0
        for run in range(runs):
            inst = make_instance(n, 3, 2 * n + 2, seed=100 * n + run)   # identical for all methods
            problem = build_csp(inst)
            size = len(problem.variables)
            for m in modes:
                r = csp_mod.solve(problem, m, node_limit=CSP_NODE_LIMIT)
                d = data[m]
                d["nodes"].append(r.nodes); d["bt"].append(r.backtracks)
                d["ms"].append(1000 * r.seconds); d["st"].append(r.status)
        for m in modes:
            d = data[m]
            star = "*" if "limit" in d["st"] else ""
            rows.append([n, size, m, "%.1f%s" % (mean(d["nodes"]), star), "%.1f%s" % (mean(d["bt"]), star),
                         "%.1f" % mean(d["ms"]),
                         "%d/%d/%d" % (d["st"].count("solved"), d["st"].count("unsatisfiable"), d["st"].count("limit"))])
    show("4. Station-scheduling CSP (%d instances per size, node cap %d; * = includes capped runs)"
         % (runs, CSP_NODE_LIMIT), header, rows, tag)
    write_csv("csp_scheduling.csv", header, rows)


def main():
    quick = "--quick" in sys.argv
    dataset = load_dataset(download=False)
    tag = TAG[0] = source_tag(dataset)
    train, test = stratified_split(dataset.plates, seed=42)
    th = Thresholds(train)
    print("Benchmarks on %s  (train=%d, test=%d)" % (tag, len(train), len(test)))
    bench_classifier(train, test, th, tag)
    bench_robust(test, th, tag, test[:60] if quick else None)
    bench_search(tag, 1 if quick else RUNS, 15 if quick else PLATES_PER_K)
    bench_csp(tag, 1 if quick else RUNS)
    print("\nCSV files written to %s/ (%s)" % (OUT, tag))


if __name__ == "__main__":
    main()
