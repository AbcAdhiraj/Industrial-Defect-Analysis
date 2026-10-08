"""Module 1 - station scheduling as a CSP (uses csp.py).

Variables : one per (plate, check) of a batch of plates.
Domains   : (station, time slot) pairs, restricted to stations that carry the
            sensor the check needs.
Constraints (all binary):
  1. a station runs one check per slot           -> two checks never share (station, slot)
  2. two checks needing the SAME sensor type cannot share a time slot
  3. two checks of the SAME plate cannot share a time slot (one plate, one place)
  4. precedence inside a plate: A in a strictly earlier slot than B
Whether stations are equipped with which sensors is a MODELLING CHOICE.
"""
import random
from collections import namedtuple

from csp import CSP
from kb import CHECKS, FAULT_INFO, MANDATORY_CHECK
from routing import precedence_for

SENSORS = sorted(set(CHECKS.values()))
Instance = namedtuple("Instance", "plates equipment n_slots")   # plates: [(pid, [checks])]


def make_equipment(n_stations, seed):
    """{station: set of sensors}; every sensor type is on at least one station
    (round-robin over a seeded shuffle, 3 sensors per station)."""
    order = list(SENSORS)
    random.Random(seed).shuffle(order)
    return {"S%d" % i: {order[(2 * i + j) % len(order)] for j in range(3)}
            for i in range(max(3, n_stations))}


def checks_for_fault(fault):
    return [MANDATORY_CHECK] + [c for c in FAULT_INFO[fault][2] if c != MANDATORY_CHECK]


def make_instance(n_plates, n_stations=3, n_slots=8, seed=0):
    """Random batch: each plate gets a random fault and hence its check list.
    Deterministic in `seed`. O(n_plates)."""
    rng = random.Random(seed)
    faults = sorted(FAULT_INFO)
    plates = [(pid, checks_for_fault(rng.choice(faults))) for pid in range(n_plates)]
    return Instance(plates, make_equipment(n_stations, seed), n_slots)


def _pair_predicate(x, y, precedence):
    """Predicate over the (station, slot) values of variables x and y."""
    same_sensor = CHECKS[x[1]] == CHECKS[y[1]]
    same_plate = x[0] == y[0]
    order = 0                                    # +1: x before y, -1: y before x
    if same_plate and (x[1], y[1]) in precedence[x[0]]:
        order = 1
    elif same_plate and (y[1], x[1]) in precedence[x[0]]:
        order = -1

    def ok(a, b):
        if a == b:                               # same station AND slot
            return False
        if (same_sensor or same_plate) and a[1] == b[1]:
            return False
        return not ((order == 1 and a[1] >= b[1]) or (order == -1 and b[1] >= a[1]))
    return ok


def build_csp(instance):
    """Turn a scheduling instance into a binary CSP. O(V^2) constraints for
    V checks, each predicate O(1)."""
    precedence = {pid: precedence_for(checks) for pid, checks in instance.plates}
    variables = [(pid, c) for pid, checks in instance.plates for c in checks]
    domains = {v: [(st, t) for st, gear in sorted(instance.equipment.items())
                   if CHECKS[v[1]] in gear for t in range(instance.n_slots)]
               for v in variables}
    csp = CSP(variables, domains)
    for i, x in enumerate(variables):
        for y in variables[i + 1:]:
            csp.add_constraint(x, y, _pair_predicate(x, y, precedence))
    return csp


def verify_schedule(instance, assignment):
    """Independent re-check of every constraint straight from the instance
    (does not use the CSP object). Returns a list of violation strings. O(V^2)."""
    bad = []
    precedence = {pid: precedence_for(checks) for pid, checks in instance.plates}
    items = list(assignment.items())
    for (var, (st, slot)) in items:
        if CHECKS[var[1]] not in instance.equipment[st]:
            bad.append("%s on %s lacks sensor" % (var, st))
        if not 0 <= slot < instance.n_slots:
            bad.append("%s slot out of range" % (var,))
    for i, (u, (su, tu)) in enumerate(items):
        for v, (sv, tv) in items[i + 1:]:
            if (su, tu) == (sv, tv):
                bad.append("%s and %s share station+slot" % (u, v))
            if CHECKS[u[1]] == CHECKS[v[1]] and tu == tv:
                bad.append("%s and %s share sensor in slot %d" % (u, v, tu))
            if u[0] == v[0] and tu == tv:
                bad.append("%s and %s: same plate, same slot" % (u, v))
            if u[0] == v[0]:
                if (u[1], v[1]) in precedence[u[0]] and not tu < tv:
                    bad.append("precedence %s<%s violated" % (u, v))
                if (v[1], u[1]) in precedence[u[0]] and not tv < tu:
                    bad.append("precedence %s<%s violated" % (v, u))
    return bad


def format_schedule(instance, assignment):
    """Slot x station grid; each cell is plate:check."""
    stations = sorted(instance.equipment)
    grid = {(st, t): "p%d:%s" % (v[0], v[1]) for v, (st, t) in assignment.items()}
    used = sorted({t for _, t in grid})
    lines = ["slot  " + "".join("%-26s" % st for st in stations)]
    for t in used:
        lines.append("%-6d" % t + "".join("%-26s" % grid.get((st, t), "-") for st in stations))
    return "\n".join(lines)
