"""Data loading, validation and splitting for the UCI Steel Plates Faults data.

Real file : data/Faults.NNA  (1941 rows x 34 whitespace-separated columns:
            27 features, then 7 one-hot class flags).
Fallback  : if the real file is missing (and cannot be downloaded) a clearly
            labelled SYNTHETIC generator with the SAME schema is used so the
            code and tests can run. Synthetic numbers say NOTHING about how
            well the system works on real plates.
"""
import io
import math
import os
import random
import urllib.request
import zipfile
from collections import Counter, namedtuple

FEATURES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]
CLASSES = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps",
           "Other_Faults"]
N_ROWS, N_COLS = 1941, 34

# Plate = one steel plate: pid (row number), features (dict), label (class name)
Plate = namedtuple("Plate", "pid features label")
Dataset = namedtuple("Dataset", "plates source")  # source: "REAL" | "SYNTHETIC"

DOWNLOAD_URLS = [
    "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA",
]
# Class sizes used ONLY by the synthetic generator (modelled on the UCI page).
SYNTH_COUNTS = {"Pastry": 158, "Z_Scratch": 190, "K_Scatch": 391, "Stains": 72,
                "Dirtiness": 55, "Bumps": 402, "Other_Faults": 673}


# ---------------------------------------------------------------- real file
def parse_nna(path):
    """Read Faults.NNA into a list of float rows.
    Purpose: turn the whitespace file into numbers. Time O(R*C), space O(R*C):
    every character is read once. Blank lines are skipped."""
    with open(path) as handle:
        return [[float(tok) for tok in line.split()]
                for line in handle if line.strip()]


def validate_matrix(matrix, expected_rows=N_ROWS):
    """Check shape and one-hot class flags; return Counter of class sizes.
    Time O(R*C). WHY: a wrong delimiter or a truncated download would
    silently give wrong science, so we fail loudly with a clear message."""
    if len(matrix) != expected_rows:
        raise ValueError("expected %d rows, found %d" % (expected_rows, len(matrix)))
    counts = Counter()
    for i, row in enumerate(matrix):
        if len(row) != N_COLS:
            raise ValueError("row %d has %d columns, expected %d" % (i, len(row), N_COLS))
        flags = row[27:]
        if sorted(flags) != [0.0] * 6 + [1.0]:
            raise ValueError("row %d does not have exactly one class flag" % i)
        counts[CLASSES[flags.index(1.0)]] += 1
    return counts


def plates_from_matrix(matrix):
    """Convert validated rows to Plate records. Time/space O(R*C)."""
    plates = []
    for pid, row in enumerate(matrix):
        features = dict(zip(FEATURES, row[:27]))
        plates.append(Plate(pid, features, CLASSES[row[27:].index(1.0)]))
    return plates


def try_download(path, timeout=15):
    """Try to fetch Faults.NNA from UCI into `path`. Returns True on success.
    Never raises: a blocked network just returns False (caller falls back)."""
    for url in DOWNLOAD_URLS:
        try:
            raw = urllib.request.urlopen(url, timeout=timeout).read()
            if raw[:2] == b"PK":                       # a zip archive
                archive = zipfile.ZipFile(io.BytesIO(raw))
                member = [n for n in archive.namelist() if n.lower().endswith("faults.nna")][0]
                raw = archive.read(member)
            validate_matrix([[float(t) for t in ln.split()]
                             for ln in raw.decode().splitlines() if ln.strip()])
            with open(path, "wb") as handle:
                handle.write(raw)
            return True
        except Exception:                              # network, zip or format error
            continue
    return False


# ---------------------------------------------------------- synthetic data
# Per-class profile: (log10 width, log10 height, fill, min lum, max lum,
# thickness, P(A300), edges, outside_x, luminosity_index, conveyer choices).
# These numbers are INVENTED for the stand-in generator; they are not UCI facts.
_PROFILES = {
    "Pastry":    (1.9, 1.7, .60, 85, 150, 60, .8, .35, .15, -.10, (1360, 1687)),
    "Z_Scratch": (2.3, 1.2, .55, 75, 150, 45, .9, .20, .05, -.15, (1360, 1687)),
    "K_Scatch":  (2.5, 2.2, .55, 60, 190, 55, .85, .45, .20, -.05, (1360, 1687, 1705)),
    "Stains":    (2.3, 2.3, .80, 110, 140, 120, .1, .60, .60, -.02, (1360, 1705)),
    "Dirtiness": (1.5, 1.4, .70, 70, 110, 100, .1, .25, .10, -.25, (1360, 1705)),
    "Bumps":     (1.5, 1.9, .60, 80, 130, 65, .6, .30, .10, -.12, (1360, 1687)),
}


def _clip(value, low, high):
    return max(low, min(high, value))


def _synthetic_row(label, rng):
    """Draw one synthetic plate (34 numbers). O(1).
    Related features are computed from the same width/height/area so that,
    like the real data, several columns carry redundant information."""
    name = label
    if label == "Other_Faults":                        # broad, overlapping mixture
        name = rng.choice(sorted(_PROFILES))
    lw, lh, fill, lmin, lmax, thick, a300, edges, outx, lumi, conv = _PROFILES[name]
    jitter = 0.35 if label == "Other_Faults" else 0.0
    if label == "Other_Faults" and rng.random() < 0.3:  # wholly unusual plate
        lw, lh, fill = rng.uniform(1, 2.7), rng.uniform(1, 2.5), rng.uniform(.2, .95)
        thick, edges = rng.uniform(40, 200), rng.uniform(0, 1)
    sd = 0.30 + jitter
    width = 10 ** rng.gauss(lw, sd)
    height = 10 ** rng.gauss(lh, sd)
    area = max(2.0, _clip(rng.gauss(fill, .12), .1, 1) * width * height)
    x_min, y_min = rng.uniform(0, 1700), rng.uniform(0, 1.3e6)
    lo_lum = _clip(rng.gauss(lmin, 15), 0, 200)
    hi_lum = _clip(lo_lum + max(5, rng.gauss(lmax - lmin, 20)), 0, 255)
    a300_flag = 1.0 if rng.random() < a300 else 0.0
    thickness = _clip(round(math.exp(rng.gauss(math.log(thick), .3))), 40, 300)
    row = [x_min, x_min + width, y_min, y_min + height, round(area),
           max(1, round(width * rng.uniform(1.0, 1.6))),
           max(1, round(height * rng.uniform(1.0, 1.6))),
           area * (lo_lum + hi_lum) / 2 * rng.uniform(.9, 1.1), lo_lum, hi_lum,
           rng.choice(conv), a300_flag, 1.0 - a300_flag, thickness,
           _clip(rng.gauss(edges, .15), 0, 1),
           _clip(1 - fill + rng.gauss(0, .08), 0, 1),
           _clip(min(width, height) / max(width, height) * rng.uniform(.8, 1.1), 0, 1),
           _clip(rng.gauss(outx, .1), 0, 1), _clip(rng.gauss(edges, .2), 0, 1),
           _clip(rng.gauss(edges, .2), 0, 1),
           rng.choices([0.0, .5, 1.0], [1 - outx - .1, .1, outx])[0],
           math.log10(area), math.log10(max(width, 1.0)), math.log10(max(height, 1.0)),
           _clip((width - height) / (width + height) + rng.gauss(0, .05), -1, 1),
           _clip(rng.gauss(lumi, .08), -1, 1),
           1 / (1 + math.exp(-(math.log10(area) - 2) * 1.5))]
    flags = [1.0 if c == label else 0.0 for c in CLASSES]
    return row + flags


def make_synthetic_matrix(seed=2024, counts=None):
    """Whole synthetic data set as a 34-column matrix, in shuffled order.
    Deterministic for a given seed. O(R*C)."""
    rng = random.Random(seed)
    matrix = [_synthetic_row(label, rng)
              for label, n in (counts or SYNTH_COUNTS).items() for _ in range(n)]
    rng.shuffle(matrix)
    return matrix


# ------------------------------------------------------------------ public
def load_dataset(data_dir="data", download=True, seed=2024):
    """Load the real file if present (or downloadable), else synthetic data.
    Returns Dataset(plates, source). Always validates with `validate_matrix`.
    The real file is NEVER overwritten by synthetic data."""
    path = os.path.join(data_dir, "Faults.NNA")
    if not os.path.exists(path) and download:
        os.makedirs(data_dir, exist_ok=True)
        try_download(path)
    if os.path.exists(path):
        matrix, source = parse_nna(path), "REAL"
    else:
        matrix, source = make_synthetic_matrix(seed), "SYNTHETIC"
    validate_matrix(matrix)
    return Dataset(plates_from_matrix(matrix), source)


def class_counts(plates):
    """Counter of class sizes, in CLASSES order for printing. O(n)."""
    counts = Counter(p.label for p in plates)
    return {c: counts.get(c, 0) for c in CLASSES}


def stratified_split(plates, train_frac=0.7, seed=42):
    """Split into (train, test) keeping class proportions.
    Purpose: every class appears in both halves (the rare Dirtiness class
    would otherwise vanish from a small test set). Time O(n log n) for the
    shuffles, space O(n). WHY it is leak-free: the split depends only on the
    seed and class sizes, never on feature values."""
    rng = random.Random(seed)
    train, test = [], []
    for label in CLASSES:
        members = [p for p in plates if p.label == label]
        rng.shuffle(members)
        cut = round(len(members) * train_frac)
        train += members[:cut]
        test += members[cut:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def banner(dataset):
    """One-line label that every printed result must carry."""
    if dataset.source == "REAL":
        return "DATA: REAL UCI Steel Plates Faults (data/Faults.NNA)"
    return ("*** SYNTHETIC DATA *** (data/Faults.NNA not found - place the real file "
            "there; results below do NOT describe real steel plates)")


def describe(dataset, train, test):
    """Printable verification summary (rows, columns, class counts, split)."""
    lines = [banner(dataset),
             "rows=%d columns=%d (27 features + 7 class flags), one class flag per row: OK"
             % (len(dataset.plates), N_COLS), "class counts:"]
    lines += ["  %-13s %4d" % kv for kv in class_counts(dataset.plates).items()]
    lines.append("stratified split (seed 42): train=%d test=%d" % (len(train), len(test)))
    return "\n".join(lines)
