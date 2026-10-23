"""Scoring logic: bin lookup from raw values, risk classification, input validation."""
import bisect
import json
from pathlib import Path
from typing import Iterable

import yaml

BASE = Path(__file__).parent

# Same shape as the `risk.bands` block in validation_rules.yaml.
# A score gets the first band whose "below" it is under; "below": None is the top band.
DEFAULT_BANDS = [
    {"name": "Very high risk", "below": 570},
    {"name": "High risk", "below": 590},
    {"name": "Moderate risk", "below": 610},
    {"name": "Low risk", "below": 640},
    {"name": "Very low risk", "below": None},
]


def classify_score(score: float, bands: list = DEFAULT_BANDS) -> str:
    """Map a score to one of five risk categories."""
    for band in bands:
        if band["below"] is None or score < band["below"]:
            return band["name"]
    return bands[-1]["name"]


def load_artifacts(base: Path = BASE):
    sc = json.loads((base / "scorecard.json").read_text())
    rules = yaml.safe_load((base / "validation_rules.yaml").read_text())
    return sc, rules


def build_lookup(scorecard: dict) -> dict:
    """Build lookup from the simplified scorecard JSON."""
    cat_lookup, num_lookup = {}, {}

    for var, spec in scorecard["variables"].items():
        if spec["type"] == "categorical":
            cat_lookup[var] = spec["bins"]
        else:
            bins = spec["bins"]
            uppers = [float("inf") if hi is None else hi for _, hi, _ in bins]
            points = [pts for _, _, pts in bins]
            num_lookup[var] = {
                "uppers": uppers,
                "points": points,
                "missing": spec.get("missing"),
            }

    return {
        "categorical": cat_lookup,
        "numeric": num_lookup,
        "offset": scorecard["offset"],
        "variables": list(scorecard["variables"].keys()),
        "types": {v: s["type"] for v, s in scorecard["variables"].items()},
    }


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _score_row(row: dict, lookup: dict) -> tuple[float, dict]:
    total = lookup["offset"]
    breakdown = {}

    for var in lookup["variables"]:
        if var not in row:
            raise KeyError(f"Missing input variable: {var}")
        raw = row[var]

        if lookup["types"][var] == "categorical":
            try:
                pts = lookup["categorical"][var][raw]
            except KeyError:
                raise ValueError(f"Unknown category {raw!r} for '{var}'")
        else:
            info = lookup["numeric"][var]
            if _is_missing(raw):
                if info["missing"] is None:
                    raise ValueError(f"No missing bin defined for '{var}'")
                pts = info["missing"]
            else:
                idx = bisect.bisect_right(info["uppers"], raw)
                idx = min(idx, len(info["points"]) - 1)
                pts = info["points"][idx]

        breakdown[var] = pts
        total += pts

    return total, breakdown


def score_dataset(data: Iterable[dict], lookup: dict, return_breakdown: bool = False):
    """Score raw-value records. Returns scores, or (scores, per-variable points)."""
    if hasattr(data, "to_dict"):
        records = data.to_dict(orient="records")
    else:
        records = list(data)

    scores, breakdowns = [], []
    for i, row in enumerate(records):
        try:
            score, parts = _score_row(row, lookup)
        except Exception as e:
            raise type(e)(f"Row {i}: {e}") from e
        scores.append(score)
        if return_breakdown:
            breakdowns.append(parts)

    return (scores, breakdowns) if return_breakdown else scores


def validate_inputs(record: dict, rules: dict, sc: dict):
    """Check one borrower's inputs against validation_rules.yaml. Returns (errors, warnings)."""
    errors, warnings = [], []
    for var, r in rules["inputs"].items():
        def msg(kind):
            return rules["messages"][kind].format(label=r["label"], min=r.get("min"), max=r.get("max"))

        v = record.get(var)
        if v is None:
            if r.get("required", True) and not r.get("allow_missing"):
                errors.append(msg("required"))
            continue
        if r["type"] == "categorical":
            if v not in sc["variables"][var]["bins"]:
                errors.append(msg("invalid_value"))
            continue
        if r.get("min") is not None and v < r["min"]:
            errors.append(msg("below_min"))
        if r.get("max") is not None and v > r["max"]:
            errors.append(msg("above_max"))
        if r.get("integer") and v != int(v):
            errors.append(msg("not_integer"))
        if (r.get("soft_min") is not None and v < r["soft_min"]) or \
           (r.get("soft_max") is not None and v > r["soft_max"]):
            warnings.append(msg("outside_training_range"))
    return errors, warnings


def derive_inputs(record: dict, rules: dict) -> dict:
    """Add derived model variables (ratio of two inputs, e.g. loan_amnt / annual_inc)."""
    row = dict(record)
    for name, spec in rules.get("derived", {}).items():
        num, den = spec["inputs"]
        row[name] = record[num] / record[den]
    return row
