"""Evaluation of the rule-based classifier against two baselines.

All thresholds / majority choices are derived from the TRAIN split only; the
TEST split is used once, for scoring.
"""
from data_loader import CLASSES, FEATURES
from inference import classify
from kb import quantile


def accuracy(true, predicted):
    return sum(t == p for t, p in zip(true, predicted)) / len(true)


def confusion_matrix(true, predicted):
    """7x7 list of lists; row = true class, column = predicted class. O(n)."""
    index = {c: i for i, c in enumerate(CLASSES)}
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    for t, p in zip(true, predicted):
        matrix[index[t]][index[p]] += 1
    return matrix


def format_confusion(matrix):
    head = "%-13s" % "true\\pred" + "".join("%7s" % c[:6] for c in CLASSES) + "   recall"
    rows = [head]
    for cls, row in zip(CLASSES, matrix):
        total = sum(row)
        rows.append("%-13s" % cls + "".join("%7d" % v for v in row)
                    + "   %5.1f%%" % (100.0 * row[CLASSES.index(cls)] / total if total else 0))
    return "\n".join(rows)


def majority_baseline(train):
    """Baseline (a): always predict the most common TRAIN class. O(n)."""
    counts = {c: 0 for c in CLASSES}
    for p in train:
        counts[p.label] += 1
    return max(CLASSES, key=lambda c: counts[c])


def _majority(plates):
    counts = {}
    for p in plates:
        counts[p.label] = counts.get(p.label, 0) + 1
    return max(sorted(counts), key=lambda c: counts[c]) if counts else CLASSES[-1]


def best_single_feature_rule(train, steps=19):
    """Baseline (b): the best rule 'IF feature <= t THEN class A ELSE class B'.

    Purpose: show how much a trivial one-feature rule achieves, so the expert
    system has something honest to beat. For every feature and every
    candidate t (the 5%..95% train quantiles) A / B are the majority train
    classes on each side; we keep the (feature, t) with best TRAIN accuracy.
    Time O(F * steps * n), space O(n). WHY quantile thresholds: they cover the
    feature range evenly while keeping the search finite.
    """
    best = None
    for name in FEATURES:
        ordered = sorted(p.features[name] for p in train)
        for k in range(1, steps + 1):
            t = quantile(ordered, k / (steps + 1))
            left = [p for p in train if p.features[name] <= t]
            right = [p for p in train if p.features[name] > t]
            if not left or not right:
                continue
            a, b = _majority(left), _majority(right)
            hits = sum(p.label == a for p in left) + sum(p.label == b for p in right)
            if best is None or hits > best[0]:
                best = (hits, name, t, a, b)
    hits, name, t, a, b = best
    return {"feature": name, "threshold": t, "left": a, "right": b,
            "train_accuracy": hits / len(train)}


def apply_single_feature(rule, plate):
    return rule["left"] if plate.features[rule["feature"]] <= rule["threshold"] else rule["right"]


def evaluate_all(train, test, thresholds):
    """Score the expert system and both baselines on `test`. Returns a dict."""
    true = [p.label for p in test]
    predicted = [classify(p.features, thresholds) for p in test]
    major = majority_baseline(train)
    single = best_single_feature_rule(train)
    return {
        "expert_accuracy": accuracy(true, predicted),
        "majority_class": major,
        "majority_accuracy": accuracy(true, [major] * len(test)),
        "single_rule": single,
        "single_accuracy": accuracy(true, [apply_single_feature(single, p) for p in test]),
        "confusion": confusion_matrix(true, predicted),
        "predicted": predicted,
    }
