"""Unit tests for metrics.py verifying normalization, edit distance, and F1 scoring."""

from __future__ import annotations

import pytest
from metrics import (
    evaluate_prediction,
    levenshtein_distance,
    norm_robust,
    norm_strict,
    normalized_edit_distance,
)


def test_norm_strict():
    assert norm_strict("  Hello   World  ") == "hello world"
    assert norm_strict("MR D.I.Y.") == "mr d.i.y."
    assert norm_strict(None) == ""


def test_norm_robust_currencies_and_numbers():
    # Malaysian Ringgit currency symbol
    assert norm_robust("RM 33.90") == "33.9"
    assert norm_robust("33.90") == "33.9"
    assert norm_robust("33.9") == "33.9"

    # Indian number commas and rupee symbol
    assert norm_robust("88,00,000") == "8800000"
    assert norm_robust("Rs. 8800000") == "8800000"
    assert norm_robust("8800000") == "8800000"

    # US dollars
    assert norm_robust("$ 1,234.50") == "1234.5"
    assert norm_robust("1234.50") == "1234.5"


def test_norm_robust_gender_and_units():
    assert norm_robust("M") == "male"
    assert norm_robust("Male") == "male"
    assert norm_robust("F") == "female"
    assert norm_robust("Female") == "female"
    assert norm_robust("19 years") == "19"
    assert norm_robust("19") == "19"


def test_norm_robust_dates():
    assert norm_robust("2019-12-01") == "2019-12-01"
    assert norm_robust("01/12/2019") == "2019-12-01"
    assert norm_robust("01-12-2019") == "2019-12-01"


def test_edit_distance_and_ned():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert normalized_edit_distance("test", "test") == 1.0
    assert normalized_edit_distance("", "") == 1.0

    # Minor OCR typo
    ned = normalized_edit_distance("Aninban Ghosh", "Anirban Ghosh")
    assert 0.90 < ned < 1.0

    # Completely different strings
    assert normalized_edit_distance("apple", "zebra") == 0.0


def test_evaluate_prediction_exact():
    truth = {"company": "Acme Corp", "total": "100.00"}
    pred = {"company": "Acme Corp", "total": "100.00"}
    result = evaluate_prediction(truth, pred)

    assert result["strict_correct"] == 2
    assert result["robust_correct"] == 2
    assert result["form_correct_strict"] is True
    assert result["form_correct_robust"] is True
    assert result["precision_strict"] == 1.0
    assert result["recall_strict"] == 1.0
    assert result["f1_strict"] == 1.0
    assert result["mean_ned"] == 1.0


def test_evaluate_prediction_robust_vs_strict():
    truth = {"total": "33.90", "sex": "Female"}
    pred = {"total": "RM 33.90", "sex": "f"}
    result = evaluate_prediction(truth, pred)

    # Strict fails because of "RM " and "f" vs "Female"
    assert result["strict_correct"] == 0
    assert result["form_correct_strict"] is False
    # Robust succeeds
    assert result["robust_correct"] == 2
    assert result["form_correct_robust"] is True
    assert result["recall_robust"] == 1.0


def test_evaluate_prediction_hallucination_penalty():
    truth = {"name": "Alice"}
    # Model returns true field plus 3 hallucinated fields
    pred = {"name": "Alice", "extra1": "foo", "extra2": "bar", "extra3": "baz"}
    result = evaluate_prediction(truth, pred)

    assert result["strict_correct"] == 1
    assert result["recall_strict"] == 1.0
    assert result["precision_strict"] == 0.25
    assert result["f1_strict"] < 0.5

