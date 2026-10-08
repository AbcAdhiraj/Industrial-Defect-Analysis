"""Module 3 - knowledge representation: discretisation, frames, semantic net.

The rule base itself lives in rules.py (it is only data); the inference
engine that runs it lives in inference.py.

MODELLING CHOICES (not facts from the data set): the fault taxonomy, the
severity levels, the check catalogue (sensors, precedence) and the
required_checks of each fault are engineering assumptions made for this
project so that the pipeline has something to plan and schedule.
"""
from collections import namedtuple

from data_loader import FEATURES, CLASSES

LEVELS = ("LOW", "MEDIUM", "HIGH")
QUANTILES = (0.25, 0.75)          # LOW <= q25 < MEDIUM <= q75 < HIGH

# A rule: IF all conditions hold THEN conclusion. A condition is
# (attribute, frozenset_of_allowed_values); conclusion is (attribute, value).
Rule = namedtuple("Rule", "rid text priority conditions conclusion why")


# ------------------------------------------------------- discretisation
def quantile(sorted_values, q):
    """Linear-interpolated quantile of an already sorted list.
    O(1) after sorting. WHY: position q*(n-1) lies between two sample
    points; we blend them so the result is smooth and needs no libraries."""
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


class Thresholds:
    """Per-feature (low, high, scale) table built ONLY from training plates.

    value <= low -> LOW ; value > high -> HIGH ; otherwise MEDIUM.
    `scale` (the training standard deviation) is used by the robustness
    check to size the sensor-noise band. Binary features get low=high=0 so
    0 -> LOW and 1 -> HIGH; they are marked non-noisy (a flag is not measured).
    """

    def __init__(self, train_plates):
        self.table, self.binary = {}, set()
        for name in FEATURES:
            values = sorted(p.features[name] for p in train_plates)
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            if set(values) <= {0.0, 1.0}:
                self.table[name] = (0.0, 0.0, 0.0)
                self.binary.add(name)
            else:
                self.table[name] = (quantile(values, QUANTILES[0]),
                                    quantile(values, QUANTILES[1]), std)

    def level(self, name, value):
        low, high, _ = self.table[name]
        return "LOW" if value <= low else ("HIGH" if value > high else "MEDIUM")

    def discretise(self, features):
        """Numeric feature dict -> {feature: LOW/MEDIUM/HIGH}. O(F)."""
        return {name: self.level(name, features[name]) for name in FEATURES}

    def noise_delta(self, name, noise_frac):
        """Sensor-noise half-width for a feature (0 for binary flags)."""
        return 0.0 if name in self.binary else noise_frac * self.table[name][2]

    def render(self):
        """The one table of thresholds, printable."""
        lines = ["%-24s %14s %14s %12s" % ("feature", "LOW <=", "HIGH >", "train std")]
        for name in FEATURES:
            low, high, std = self.table[name]
            lines.append("%-24s %14.4f %14.4f %12.4f" % (name, low, high, std))
        return "\n".join(lines)


# ------------------------------------------------------------- frames
class Frame:
    """A frame: a named bundle of slots with optional is-a inheritance.
    get() looks in the frame itself, then walks up `parent` frames, which is
    how defaults are inherited (e.g. every FaultType shares the 'Fault' slots).
    """

    def __init__(self, name, parent=None, **slots):
        self.name, self.parent, self.slots = name, parent, dict(slots)

    def get(self, slot, default=None):
        frame = self
        while frame is not None:
            if slot in frame.slots:
                return frame.slots[slot]
            frame = frame.parent
        return default

    def __repr__(self):
        return "Frame(%s, %s)" % (self.name, self.slots)


# Check catalogue: name -> sensor needed (used by the scheduling CSP).
CHECKS = {
    "ALIGNMENT": "LASER", "SURFACE_SCAN": "CAMERA", "EDGE_PROFILE": "LASER",
    "THICKNESS_GAUGE": "ULTRASONIC", "LUMINOSITY_MAP": "CAMERA",
    "WIPE_TEST": "MANUAL", "FLATNESS": "LASER", "DEPTH_PROBE": "ULTRASONIC",
    "MAGNETIC_PARTICLE": "MAGNETIC", "MANUAL_REVIEW": "MANUAL",
}
# "A before B" pairs, applied only when both checks are present. They form
# an acyclic order, so every subset of checks has a feasible visiting order.
PRECEDENCE_PAIRS = [("SURFACE_SCAN", "DEPTH_PROBE"), ("EDGE_PROFILE", "THICKNESS_GAUGE"),
                    ("WIPE_TEST", "LUMINOSITY_MAP"), ("FLATNESS", "MANUAL_REVIEW"),
                    ("DEPTH_PROBE", "MAGNETIC_PARTICLE"), ("THICKNESS_GAUGE", "MANUAL_REVIEW")]
MANDATORY_CHECK = "ALIGNMENT"

FAULT_INFO = {   # fault -> (severity, typical conditions, required_checks)
    "Pastry": ("MEDIUM", "medium patch, bright, thin plate",
               ["SURFACE_SCAN", "LUMINOSITY_MAP", "EDGE_PROFILE"]),
    "Z_Scratch": ("HIGH", "long thin mark along X, thin plate",
                  ["SURFACE_SCAN", "EDGE_PROFILE", "THICKNESS_GAUGE"]),
    "K_Scatch": ("HIGH", "large bright-sum area, wide in X and Y",
                 ["SURFACE_SCAN", "DEPTH_PROBE", "THICKNESS_GAUGE"]),
    "Stains": ("LOW", "large filled patch on thick A400 plate",
               ["LUMINOSITY_MAP", "WIPE_TEST", "SURFACE_SCAN"]),
    "Dirtiness": ("LOW", "small dark patch on thick A400 plate",
                  ["WIPE_TEST", "LUMINOSITY_MAP"]),
    "Bumps": ("MEDIUM", "tall narrow mark, negative orientation",
              ["FLATNESS", "THICKNESS_GAUGE", "EDGE_PROFILE", "SURFACE_SCAN"]),
    "Other_Faults": ("MEDIUM", "no specific pattern (fallback)",
                     ["SURFACE_SCAN", "FLATNESS", "THICKNESS_GAUGE", "EDGE_PROFILE",
                      "MANUAL_REVIEW"]),
}


def build_fault_frames():
    """FaultType frames (all children of one generic 'Fault' frame).
    Slots: typical_conditions, severity, required_checks. O(#faults)."""
    root = Frame("Fault", severity="MEDIUM", required_checks=[MANDATORY_CHECK])
    return {name: Frame(name, root, severity=info[0], typical_conditions=info[1],
                        required_checks=list(info[2]))
            for name, info in FAULT_INFO.items()}


def plate_frame(pid, facts):
    """Plate frame whose slots are the discretised features. O(F)."""
    return Frame("Plate_%d" % pid, None, **facts)


# --------------------------------------------------------- semantic net
class SemanticNet:
    """Labelled directed graph of (subject, relation, object) triples.

    is_a() is a graph search over 'is-a' links (transitive closure on demand);
    inherited() lets a property stated high in the hierarchy answer a question
    about a leaf. Time O(V+E) per query, space O(V).
    """

    def __init__(self):
        self.edges = []

    def add(self, subject, relation, obj):
        self.edges.append((subject, relation, obj))

    def related(self, subject, relation):
        return [o for s, r, o in self.edges if s == subject and r == relation]

    def ancestors(self, node):
        """Everything reachable by following is-a links upwards, nearest first."""
        found, frontier = [], [node]
        while frontier:
            nxt = []
            for current in frontier:
                for parent in self.related(current, "is-a"):
                    if parent not in found:
                        found.append(parent)
                        nxt.append(parent)
            frontier = nxt
        return found

    def is_a(self, node, category):
        return node == category or category in self.ancestors(node)

    def members(self, category):
        """All nodes below `category` (inverse query)."""
        return sorted(s for s, r, _ in self.edges if r == "is-a" and self.is_a(s, category)
                      and s != category)

    def inherited(self, node, relation):
        """First value of `relation` on the node or its nearest ancestor."""
        for current in [node] + self.ancestors(node):
            values = self.related(current, relation)
            if values:
                return values[0]
        return None


def build_semantic_net():
    """Fault taxonomy (a MODELLING CHOICE, see file header)."""
    net = SemanticNet()
    for child, parent in [("Z_Scratch", "SurfaceDefect"), ("K_Scatch", "SurfaceDefect"),
                          ("Stains", "ContaminationDefect"), ("Dirtiness", "ContaminationDefect"),
                          ("Bumps", "ShapeDefect"), ("Pastry", "ShapeDefect"),
                          ("SurfaceDefect", "Fault"), ("ContaminationDefect", "Fault"),
                          ("ShapeDefect", "Fault"), ("Other_Faults", "Fault")]:
        net.add(child, "is-a", parent)
    net.add("SurfaceDefect", "detected-by", "CAMERA")
    net.add("ContaminationDefect", "detected-by", "MANUAL")
    net.add("ShapeDefect", "detected-by", "LASER")
    net.add("Fault", "detected-by", "CAMERA")
    assert all(c in dict.fromkeys(s for s, _, _ in net.edges) for c in CLASSES)
    return net
