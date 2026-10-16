"""Unit tests for schemas.py ensuring vLLM strict JSON schema compliance."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from schemas import (
    FIXED_SCHEMAS,
    SROIEInvoice,
    NigeriaMedicalRecord,
    IDRBTCheque,
    IAMHandwriting,
    create_template_model,
    get_schema_for_example,
    sanitize_field_name,
    to_vllm_json_schema,
)


def test_fixed_schemas_exist():
    assert "SROIE_Invoice Dataset" in FIXED_SCHEMAS
    assert "Nigeria_medical_records" in FIXED_SCHEMAS
    assert "IDRBT_Cheque_Dataset" in FIXED_SCHEMAS
    assert "IAM handwriting finevision" in FIXED_SCHEMAS


@pytest.mark.parametrize("dataset_name,model_cls", FIXED_SCHEMAS.items())
def test_fixed_schema_vllm_compliance(dataset_name: str, model_cls: type[BaseModel]):
    schema = to_vllm_json_schema(model_cls)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "properties" in schema
    assert "required" in schema
    # All properties must be in required in strict mode
    assert set(schema["required"]) == set(schema["properties"].keys())
    # Titles should be removed
    assert "title" not in schema


def test_sanitize_field_name():
    assert sanitize_field_name("Date") == "Date"
    assert sanitize_field_name("(DOB) Date") == "DOB_Date"
    assert sanitize_field_name("Father's Surname") == "Father_s_Surname"
    assert sanitize_field_name("1. Matric") == "f_1_Matric"


def test_create_template_model_with_special_characters():
    fields = [
        "Personal Information",
        "(DOB) Date",
        "Father's Surname",
        "Total fee deposited by the candidate Rs.",
    ]
    model = create_template_model("TestForm", fields)
    schema = to_vllm_json_schema(model)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(fields)

    # Test validation and dump by alias
    data = {
        "Personal Information": "John Doe",
        "(DOB) Date": "1990-01-01",
        "Father's Surname": "Doe",
        "Total fee deposited by the candidate Rs.": "5000",
    }
    instance = model.model_validate(data)
    dumped = instance.model_dump(by_alias=True)
    assert dumped == data


def test_get_schema_for_example_modes():
    # Fixed schema in template mode
    model, response_fmt, fields = get_schema_for_example("SROIE_Invoice Dataset", ["company", "date", "address", "total"], mode="template")
    assert model is SROIEInvoice
    assert response_fmt["type"] == "json_schema"
    assert response_fmt["json_schema"]["strict"] is True
    assert set(fields) == {"company", "date", "address", "total"}

    # Grouped template in template mode
    truth_fields = ["DATE", "TO", "FROM", "FAX NUMBER"]
    model2, response_fmt2, fields2 = get_schema_for_example("funds_handwritten_dataset", truth_fields, mode="template")
    assert response_fmt2["type"] == "json_schema"
    assert response_fmt2["json_schema"]["strict"] is True
    assert fields2 == truth_fields

    # Open mode
    model3, response_fmt3, fields3 = get_schema_for_example("Forms600", truth_fields, mode="open")
    assert model3 is None
    assert response_fmt3 == {"type": "json_object"}
    assert fields3 == truth_fields

