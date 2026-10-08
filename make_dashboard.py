"""Build docs/dashboard.html from the CSVs in results/ (run after benchmarks.py).

The dashboard is one self-contained HTML file. All numbers come from
results/*.csv and results/test_output.txt; the example route is recomputed
from the project code. Usage: python3 make_dashboard.py
"""
import csv
import json
import re

from data_loader import load_dataset
from pipeline import Pipeline
from routing import InspectionRoute, make_checkpoints, make_heuristics
from search import a_star, breadth_first


def read(name):
    with open("results/" + name, newline="") as handle:
        return list(csv.reader(handle))


def metric_number(text):
    return float(re.match(r"[-\d.]+", text).group())


def route_example(pipe):
    """First sampled plate with >= 5 checks: A* route versus BFS route."""
    for plate in pipe.sample(12, 0):
        report = pipe.run_plate(plate)
        if len(report.checks) >= 5:
            problem = InspectionRoute(make_checkpoints(report.checks, seed=1000 + plate.pid))
            bfs = breadth_first(problem)
            return {"pid": plate.pid, "true": report.true, "label": report.label,
                    "points": problem.points, "astar": report.route, "astar_cost": report.cost,
                    "astar_nodes": report.nodes, "bfs": bfs.path, "bfs_cost": bfs.cost,
                    "bfs_nodes": bfs.nodes_expanded,
                    "decision": report.robust.move, "fired": report.fired}
    return None


def main():
    acc = {r[0]: r for r in read("classifier_accuracy.csv")[1:]}
    conf = read("confusion_matrix.csv")
    robust = {r[0]: r[1] for r in read("robust_check.csv")[1:]}
    search = [dict(zip(["K", "method", "nodes", "ms", "ratio", "optimal", "capped"], r[:7]))
              for r in read("search_routes.csv")[1:]]
    csp = [dict(zip(["plates", "checks", "method", "nodes", "backtracks", "ms", "status"], r[:7]))
           for r in read("csp_scheduling.csv")[1:]]
    tests = [{"name": m.group(2), "ok": m.group(1) == "PASS", "checks": int(m.group(3))}
             for m in (re.match(r"\[(PASS|FAIL)\] (\S+)\s+(\d+) checks", ln)
                       for ln in open("results/test_output.txt")) if m]
    dataset = load_dataset(download=False)
    data = {"source": read("robust_check.csv")[1][-1], "classes": conf[0][1:-1],
            "accuracy": [[r[0], metric_number(r[1]), r[2]] for r in acc.values()],
            "confusion": [[int(v) for v in r[1:-1]] for r in conf[1:]],
            "robust": robust, "search": search, "csp": csp, "tests": tests,
            "route": route_example(Pipeline(dataset))}
    body = open("docs/dashboard_template.html").read().replace("__DATA__", json.dumps(data))
    page = ('<!doctype html>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + body)
    open("docs/dashboard.html", "w").write(page)
    print("wrote docs/dashboard.html (%d bytes, source: %s)" % (len(page), data["source"]))


if __name__ == "__main__":
    main()
