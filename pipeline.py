"""End-to-end pipeline: CLASSIFY -> CHECK -> PLAN -> SCHEDULE.

1 CLASSIFY (Module 3)  expert system names the fault and the rules that fired
2 CHECK    (Module 2)  adversarial search: is that label stable under noise?
3 PLAN     (Module 1)  A* orders the required inspection checkpoints
4 SCHEDULE (Module 1)  CSP assigns (station, slot) to every check of a batch
The fault frame's `required_checks` slot links step 1 to steps 3 and 4. If the
robustness check says REVIEW, a MANUAL_REVIEW checkpoint is added to the route
(design choice: an uncertain label gets a human look).
"""
import random
from collections import namedtuple

from csp import solve
from data_loader import stratified_split
from inference import forward_chain
from kb import MANDATORY_CHECK, Thresholds, build_fault_frames, build_semantic_net
from robust_check import robust_decision
from routing import InspectionRoute, make_checkpoints, make_heuristics
from scheduling import Instance, build_csp, format_schedule, make_equipment, verify_schedule
from search import a_star

PlateReport = namedtuple("PlateReport", "pid true label rule fired robust checks route cost nodes")


class Pipeline:
    def __init__(self, dataset, split_seed=42, noise_frac=0.10):
        self.dataset, self.noise_frac = dataset, noise_frac
        self.train, self.test = stratified_split(dataset.plates, seed=split_seed)
        self.thresholds = Thresholds(self.train)       # train quantiles only
        self.frames, self.net = build_fault_frames(), build_semantic_net()

    def sample(self, n, seed=0):
        return random.Random(seed).sample(self.test, n)

    def run_plate(self, plate):
        """Run steps 1-3 for one plate. The true label is carried for display
        only; no step reads it."""
        facts = self.thresholds.discretise(plate.features)
        result = forward_chain(facts)
        robust = robust_decision(plate.features, self.thresholds, self.noise_frac)
        checks = list(self.frames[result.label].get("required_checks"))
        if MANDATORY_CHECK not in checks:
            checks.insert(0, MANDATORY_CHECK)
        if robust.decision == "REVIEW" and "MANUAL_REVIEW" not in checks:
            checks.append("MANUAL_REVIEW")
        problem = InspectionRoute(make_checkpoints(checks, seed=1000 + plate.pid))
        plan = a_star(problem, make_heuristics(problem)[0])
        return PlateReport(plate.pid, plate.label, result.label, result.winner.rid,
                           [r.rid for r in result.fired], robust, checks,
                           plan.path, plan.cost, plan.nodes_expanded)

    def schedule(self, reports, n_stations=3, seed=0, node_limit=200000):
        """Step 4: CSP schedule for the whole batch. Returns (instance, CSPResult)."""
        plates = [(r.pid, r.checks) for r in reports]
        instance = Instance(plates, make_equipment(n_stations, seed),
                            3 * max(len(c) for _, c in plates) + 2)
        result = solve(build_csp(instance), "ac3_fc_mrv", node_limit=node_limit)
        if result.solutions:
            assert not verify_schedule(instance, result.solutions[0])
        return instance, result


def format_report(r):
    rb = r.robust
    lines = ["Plate %d   true=%s   predicted=%s   %s"
            % (r.pid, r.true, r.label, "(correct)" if r.true == r.label else "(WRONG)"),
            "  fired rules : %s   (winner %s)" % (", ".join(r.fired), r.rule),
            "  robustness  : %s  [game value %+.1f, borderline: %s, nodes minimax=%d alpha-beta=%d]"
            % (rb.move, rb.value, ", ".join(rb.borderline) or "none", rb.minimax_nodes, rb.alphabeta_nodes),
            "  checks      : " + ", ".join(r.checks),
            "  route (A*)  : DOCK -> " + " -> ".join(r.route) + "   cost %.1f (%d nodes)" % (r.cost, r.nodes)]
    return "\n".join(lines)


def format_batch(instance, result):
    if not result.solutions:
        return "Batch schedule: %s (nodes=%d backtracks=%d)" % (result.status.upper(), result.nodes, result.backtracks)
    return ("Batch schedule (%d checks, %d stations, %d slots; nodes=%d backtracks=%d):\n%s"
            % (len(result.solutions[0]), len(instance.equipment), instance.n_slots,
               result.nodes, result.backtracks, format_schedule(instance, result.solutions[0])))
