import math

import numpy as np


def wilson(correct, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - half, center + half]


def policy_stats(probabilities, truth, classes, policy):
    confidence = probabilities.max(axis=1)
    accepted = (
        confidence >= policy["threshold"]
        if policy["auto_routing_enabled"]
        else np.zeros(len(confidence), dtype=bool)
    )
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    n = int(accepted.sum())
    correct = int(((predictions == np.asarray(truth)) & accepted).sum())
    return {
        "n": len(truth),
        "accepted_count": n,
        "accepted_correct": correct,
        "coverage": n / len(truth),
        "manual_review_rate": 1 - n / len(truth),
        "accepted_accuracy": correct / n if n else None,
        "accepted_accuracy_wilson95": wilson(correct, n),
    }


def choose_policy(probabilities, truth, classes, target=0.95, min_accepted=100):
    confidence = probabilities.max(axis=1)
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    correct = predictions == np.asarray(truth)
    order = np.argsort(-confidence, kind="stable")
    scores, cumulative = confidence[order], np.cumsum(correct[order])
    rows = []
    # Include complete tie groups, matching the >= inference boundary exactly.
    for i in np.flatnonzero(np.r_[scores[:-1] != scores[1:], True]):
        n, successes = int(i + 1), int(cumulative[i])
        rows.append(
            {
                "threshold": float(scores[i]),
                "accepted_count": n,
                "accepted_correct": successes,
                "coverage": n / len(truth),
                "manual_review_rate": 1 - n / len(truth),
                "accepted_accuracy": successes / n,
                "accepted_accuracy_wilson95": wilson(successes, n),
            }
        )
    eligible = [
        r for r in rows if r["accepted_count"] >= min_accepted and r["accepted_accuracy"] >= target
    ]
    chosen = max(eligible, key=lambda x: (x["coverage"], x["threshold"])) if eligible else None
    policy = {
        "auto_routing_enabled": chosen is not None,
        "threshold": chosen["threshold"] if chosen else None,
        "target_accuracy": target,
        "min_accepted": min_accepted,
        "selection_split": "validation_policy",
        "n": len(truth),
    }
    policy["validation_metrics"] = policy_stats(probabilities, truth, classes, policy)
    return policy, rows


def route(confidence, policy):
    if not policy["auto_routing_enabled"]:
        return "manual_review", "auto_routing_disabled"
    if confidence >= policy["threshold"]:
        return "auto_route", None
    return "manual_review", "low_confidence"
