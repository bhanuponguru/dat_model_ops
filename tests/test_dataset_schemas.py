"""Tests verifying dataset discovery and schema compatibility across all 163 examples."""

from __future__ import annotations

from pathlib import Path
import pytest
from benchmark import discover
from schemas import get_schema_for_example, to_vllm_json_schema


def test_discover_examples():
    data_dir = Path("datasets/unpacked")
    assert data_dir.exists(), "datasets/unpacked must exist"

    examples, warnings = discover(data_dir)
    # The benchmark covers 163 paired examples across 6 datasets
    assert len(examples) == 163, f"Expected 163 examples, found {len(examples)}"

    datasets = {ex.dataset for ex in examples}
    expected_datasets = {
        "Forms600",
        "SROIE_Invoice Dataset",
        "Nigeria_medical_records",
        "funds_handwritten_dataset",
        "IDRBT_Cheque_Dataset",
        "IAM handwriting finevision",
    }
    assert datasets == expected_datasets


def test_schema_generation_for_all_examples():
    data_dir = Path("datasets/unpacked")
    examples, _ = discover(data_dir)

    for ex in examples:
        model, response_fmt, fields = get_schema_for_example(
            ex.dataset,
            list(ex.truth.keys()),
            mode="template",
        )
        assert response_fmt is not None
        assert response_fmt["type"] == "json_schema"
        json_schema = response_fmt["json_schema"]["schema"]
        assert json_schema["type"] == "object"
        assert json_schema["additionalProperties"] is False
        assert set(json_schema["required"]) == set(json_schema["properties"].keys())
        assert len(fields) > 0

