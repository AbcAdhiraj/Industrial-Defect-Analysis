"""Module 3 - the rule base (IF-THEN rules over discretised facts).

HAND-WRITTEN rules. Layer 1 rules (D*) derive "symptom" facts from the
discretised features; layer 2 rules (F*) conclude the fault. Priority:
3 = very specific / strong evidence, 2 = normal, 1 = weak, 0 = fallback.

The rules were written by reading TRAIN-split feature distributions per fault
(`python3 rule_evidence.py` prints them and writes docs/RULES.md, including
the per-rule precision measured on the TRAIN split). The test split was never
used. If the data source changes (e.g. real UCI file instead of the synthetic
stand-in) re-run rule_evidence.py and revise any rule with poor precision.
"""
from kb import Rule

DERIVED = {"ELONGATED_X", "TALL_NARROW", "LARGE_AREA", "SMALL_AREA", "THICK_PLATE",
           "THIN_PLATE", "DARK_CORE", "FULL_WIDTH_MARK", "SOLID_FILL"}


def _c(attr, values):
    """Condition: attribute takes one of the comma-separated values."""
    return (attr, frozenset(values.split(",")))


def _rule(rid, text, prio, conds, concl, why):
    return Rule(rid, text, prio, tuple(_c(a, v) for a, v in conds), concl, why)


def _sym(name):
    return (name, "YES")


RULES = [
    # ---- layer 1: symptoms derived from discretised measurements -------
    _rule("D1", "long in X, short in Y => ELONGATED_X", 2,
          [("Y_Perimeter", "LOW"), ("X_Perimeter", "MEDIUM,HIGH")], _sym("ELONGATED_X"),
          "a mark that is long along the rolling direction has a big X and small Y perimeter"),
    _rule("D2", "narrow in X, tall in Y => TALL_NARROW", 2,
          [("X_Perimeter", "LOW"), ("Y_Perimeter", "MEDIUM,HIGH")], _sym("TALL_NARROW"),
          "bumps are tall, narrow marks"),
    _rule("D3", "pixel area HIGH => LARGE_AREA", 2,
          [("Pixels_Areas", "HIGH")], _sym("LARGE_AREA"), "large defect surface"),
    _rule("D4", "pixel area LOW => SMALL_AREA", 2,
          [("Pixels_Areas", "LOW")], _sym("SMALL_AREA"), "small defect surface"),
    _rule("D5", "thickness HIGH => THICK_PLATE", 2,
          [("Steel_Plate_Thickness", "HIGH")], _sym("THICK_PLATE"),
          "stains and dirt show up mostly on thick plates"),
    _rule("D6", "thickness LOW => THIN_PLATE", 2,
          [("Steel_Plate_Thickness", "LOW")], _sym("THIN_PLATE"),
          "scratches are commoner on thin plate"),
    _rule("D7", "darkest pixel LOW => DARK_CORE", 2,
          [("Minimum_of_Luminosity", "LOW")], _sym("DARK_CORE"),
          "deep marks have a very dark core"),
    _rule("D8", "mark touches the plate edge band => FULL_WIDTH_MARK", 2,
          [("Outside_X_Index", "HIGH")], _sym("FULL_WIDTH_MARK"),
          "wide stains extend outside the normal X range"),
    _rule("D9", "no empty space inside the bounding box => SOLID_FILL", 2,
          [("Empty_Index", "LOW")], _sym("SOLID_FILL"), "a stain fills its bounding box"),
    # ---- layer 2: fault rules ------------------------------------------
    _rule("F1", "thick A400 plate, full-width mark => Stains", 3,
          [("THICK_PLATE", "YES"), ("FULL_WIDTH_MARK", "YES"), ("TypeOfSteel_A400", "HIGH")],
          ("fault", "Stains"), "three independent stain symptoms agree"),
    _rule("F2", "solid fill and bright minimum => Stains", 2,
          [("SOLID_FILL", "YES"), ("Minimum_of_Luminosity", "HIGH")],
          ("fault", "Stains"), "stains are uniformly bright patches"),
    _rule("F3", "thick A400 plate with large area => Stains", 2,
          [("THICK_PLATE", "YES"), ("LARGE_AREA", "YES"), ("TypeOfSteel_A400", "HIGH")],
          ("fault", "Stains"), "large patch on thick A400 plate"),
    _rule("F4", "small area on thick A400 plate => Dirtiness", 3,
          [("SMALL_AREA", "YES"), ("THICK_PLATE", "YES"), ("TypeOfSteel_A400", "HIGH")],
          ("fault", "Dirtiness"), "small marks on thick A400 plate"),
    _rule("F5", "dark luminosity index on A400, not large => Dirtiness", 2,
          [("Luminosity_Index", "LOW"), ("TypeOfSteel_A400", "HIGH"),
           ("Pixels_Areas", "LOW,MEDIUM")], ("fault", "Dirtiness"),
          "dirt darkens the surface relative to its surroundings"),
    _rule("F6", "thick plate, empty box, not large => Dirtiness", 1,
          [("THICK_PLATE", "YES"), ("Empty_Index", "MEDIUM,HIGH"),
           ("Pixels_Areas", "LOW,MEDIUM")], ("fault", "Dirtiness"),
          "separates dirt from solid stains on thick plate"),
    _rule("F7", "elongated mark on thin plate => Z_Scratch", 3,
          [("ELONGATED_X", "YES"), ("THIN_PLATE", "YES")],
          ("fault", "Z_Scratch"), "long thin mark on thin plate"),
    _rule("F8", "elongated mark with positive orientation => Z_Scratch", 2,
          [("ELONGATED_X", "YES"), ("Orientation_Index", "HIGH")],
          ("fault", "Z_Scratch"), "orientation index confirms a horizontal mark"),
    _rule("F9", "flat, low-Y mark inside normal X range => Z_Scratch", 2,
          [("Square_Index", "LOW"), ("Y_Perimeter", "LOW"), ("Outside_X_Index", "LOW")],
          ("fault", "Z_Scratch"), "very non-square, short in Y"),
    _rule("F10", "elongated mark alone => Z_Scratch", 1,
          [("ELONGATED_X", "YES")], ("fault", "Z_Scratch"), "weak: shape only"),
    _rule("F11", "large dark-core mark on A300 => K_Scatch", 3,
          [("LARGE_AREA", "YES"), ("DARK_CORE", "YES"), ("TypeOfSteel_A300", "HIGH")],
          ("fault", "K_Scatch"), "large deep scratch field on A300"),
    _rule("F12", "large area on A300, not solid fill => K_Scatch", 2,
          [("LARGE_AREA", "YES"), ("TypeOfSteel_A300", "HIGH"),
           ("Empty_Index", "MEDIUM,HIGH")], ("fault", "K_Scatch"),
          "large scratched area on A300 plate"),
    _rule("F13", "high luminosity sum and max on A300 => K_Scatch", 2,
          [("Sum_of_Luminosity", "HIGH"), ("Maximum_of_Luminosity", "HIGH"),
           ("TypeOfSteel_A300", "HIGH")], ("fault", "K_Scatch"),
          "bright, large scatter pattern"),
    _rule("F14", "large X and Y perimeters => K_Scatch", 1,
          [("X_Perimeter", "HIGH"), ("Y_Perimeter", "HIGH")],
          ("fault", "K_Scatch"), "weak: big outline in both directions"),
    _rule("F15", "tall narrow mark, low orientation => Bumps", 3,
          [("TALL_NARROW", "YES"), ("Orientation_Index", "LOW")],
          ("fault", "Bumps"), "vertical marks have a negative orientation index"),
    _rule("F16", "tall narrow, not large, not thick => Bumps", 2,
          [("TALL_NARROW", "YES"), ("Pixels_Areas", "LOW,MEDIUM"),
           ("Steel_Plate_Thickness", "LOW,MEDIUM")], ("fault", "Bumps"),
          "tall narrow marks on ordinary plate"),
    _rule("F17", "low orientation, plate not thick => Bumps", 1,
          [("Orientation_Index", "LOW"), ("Steel_Plate_Thickness", "LOW,MEDIUM")],
          ("fault", "Bumps"), "weak: orientation only"),
    _rule("F18", "squarish mark with bright minimum => Pastry", 2,
          [("Square_Index", "HIGH"), ("Minimum_of_Luminosity", "MEDIUM,HIGH"),
           ("Steel_Plate_Thickness", "LOW,MEDIUM")], ("fault", "Pastry"),
          "compact, bright patch on ordinary plate"),
    _rule("F19", "medium outline, bright minimum, medium area => Pastry", 1,
          [("X_Perimeter", "MEDIUM"), ("Y_Perimeter", "MEDIUM"),
           ("Pixels_Areas", "MEDIUM"), ("Minimum_of_Luminosity", "HIGH")],
          ("fault", "Pastry"), "weak: mid-sized, bright"),
    _rule("F20", "thick A300 plate => Other_Faults", 2,
          [("THICK_PLATE", "YES"), ("TypeOfSteel_A300", "HIGH")],
          ("fault", "Other_Faults"), "thick A300 plate matches no named pattern"),
    _rule("F21", "no other rule applies => Other_Faults", 0,
          [], ("fault", "Other_Faults"), "fallback (closed-world default)"),
]
