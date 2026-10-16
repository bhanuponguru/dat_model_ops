"""Evaluation metrics, domain-aware normalizers, and edit-distance calculations."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def norm_strict(val: Any) -> str:
    """Strict normalization: trim whitespace, collapse multiple spaces, casefold."""
    if val is None:
        return ""
    text = str(val).strip()
    return re.sub(r"\s+", " ", text).casefold()


def try_parse_number(text: str) -> float | None:
    """Attempt to parse text as a clean floating point number."""
    # Strip currency words, symbols, and commas
    cleaned = re.sub(r"(?i)\b(rm|rs|usd|eur|inr|dollars?|cents?)\b\.?", "", text)
    cleaned = re.sub(r"[$€£₹,\s]", "", cleaned)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def try_parse_date(text: str) -> str | None:
    """Attempt to canonicalize common date representations to YYYY-MM-DD."""
    cleaned = text.strip()
    date_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d-%m-%y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            # Handle 2-digit years reasonably
            if dt.year < 1950:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def norm_robust(val: Any) -> str:
    """Domain-aware normalization handling currencies, dates, numbers, and units."""
    if val is None:
        return ""
    text = str(val).strip()
    if not text:
        return ""

    lowered = text.casefold()

    # 1. Normalize gender/boolean tokens
    if lowered in {"m", "male"}:
        return "male"
    if lowered in {"f", "female"}:
        return "female"
    if lowered in {"y", "yes", "true"}:
        return "yes"
    if lowered in {"n", "no", "false"}:
        return "no"

    # 2. Check for numeric / currency match (e.g. RM 33.90 vs 33.90 vs 33.9, or 88,00,000 vs 8800000)
    num = try_parse_number(lowered)
    # Check if string after stripping currency/units is strictly numeric
    non_currency = re.sub(r"(?i)\b(rm|rs|usd|eur|inr|dollars?|cents?)\b\.?", "", lowered)
    non_currency = re.sub(r"[$€£₹,\s]", "", non_currency)
    if num is not None and (not non_currency or not re.search(r"[a-z]", non_currency)):
        if num.is_integer():
            return str(int(num))
        return f"{num:.4f}".rstrip("0").rstrip(".")

    # 3. Check for date match
    parsed_date = try_parse_date(text)
    if parsed_date:
        return parsed_date

    # 4. Strip currency symbols and common unit noise (e.g. "19 years" vs "19")
    cleaned = re.sub(r"(?i)\b(rm|rs\.?|usd|eur|inr|years?|yrs?)\b", "", lowered)
    cleaned = re.sub(r"[$€£₹,]", "", cleaned)
    # Collapse whitespace and punctuation padding
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[.:;\-]+|[.:;\-]+$", "", cleaned)
    return cleaned


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings using dynamic programming."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute Normalized Edit Distance (similarity) between 0.0 and 1.0.
    
    1.0 means exact string identity, 0.0 means completely disjoint.
    """
    s1_norm = norm_strict(s1)
    s2_norm = norm_strict(s2)
    if not s1_norm and not s2_norm:
        return 1.0
    max_len = max(len(s1_norm), len(s2_norm))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1_norm, s2_norm)
    return max(0.0, 1.0 - (dist / max_len))


def evaluate_prediction(
    truth: dict[str, str],
    prediction: dict[str, str],
) -> dict[str, Any]:
    """Evaluate a predicted dictionary against ground truth.
    
    Returns detailed per-field match indicators, counts, and calculated metrics.
    """
    field_details: dict[str, dict[str, Any]] = {}
    strict_correct = 0
    robust_correct = 0
    total_ned = 0.0

    all_fields = list(dict.fromkeys([*truth, *prediction]))
    truth_count = len(truth)
    pred_count = len(prediction)

    for field in all_fields:
        t_val = truth.get(field, "")
        p_val = prediction.get(field, "")
        in_truth = field in truth
        in_pred = field in prediction

        s_match = bool(in_truth and in_pred and norm_strict(t_val) == norm_strict(p_val))
        r_match = bool(in_truth and in_pred and norm_robust(t_val) == norm_robust(p_val))
        ned = normalized_edit_distance(t_val, p_val) if in_truth else 0.0

        if s_match:
            strict_correct += 1
        if r_match:
            robust_correct += 1
        if in_truth:
            total_ned += ned

        field_details[field] = {
            "truth": t_val,
            "prediction": p_val,
            "in_truth": in_truth,
            "in_pred": in_pred,
            "strict_match": s_match,
            "robust_match": r_match,
            "ned_similarity": round(ned, 4),
        }

    # Precision, Recall, F1 (Strict)
    prec_strict = (strict_correct / pred_count) if pred_count > 0 else 0.0
    rec_strict = (strict_correct / truth_count) if truth_count > 0 else 0.0
    f1_strict = (
        (2 * prec_strict * rec_strict / (prec_strict + rec_strict))
        if (prec_strict + rec_strict) > 0
        else 0.0
    )

    # Precision, Recall, F1 (Robust)
    prec_robust = (robust_correct / pred_count) if pred_count > 0 else 0.0
    rec_robust = (robust_correct / truth_count) if truth_count > 0 else 0.0
    f1_robust = (
        (2 * prec_robust * rec_robust / (prec_robust + rec_robust))
        if (prec_robust + rec_robust) > 0
        else 0.0
    )

    mean_ned = (total_ned / truth_count) if truth_count > 0 else 0.0

    return {
        "truth_count": truth_count,
        "pred_count": pred_count,
        "strict_correct": strict_correct,
        "robust_correct": robust_correct,
        "form_correct_strict": bool(truth_count > 0 and strict_correct == truth_count),
        "form_correct_robust": bool(truth_count > 0 and robust_correct == truth_count),
        "precision_strict": round(prec_strict, 4),
        "recall_strict": round(rec_strict, 4),
        "f1_strict": round(f1_strict, 4),
        "precision_robust": round(prec_robust, 4),
        "recall_robust": round(rec_robust, 4),
        "f1_robust": round(f1_robust, 4),
        "mean_ned": round(mean_ned, 4),
        "field_details": field_details,
    }
