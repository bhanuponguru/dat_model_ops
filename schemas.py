"""Pydantic schemas and vLLM-compatible JSON schema generators for form extraction."""

from __future__ import annotations

import re
from typing import Any, Type
from pydantic import BaseModel, ConfigDict, Field, create_model


class StrictExtractionModel(BaseModel):
    """Base model enforcing no extra keys and explicit nullability."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


# ==============================================================================
# Tier 1: Fixed Schemas (All documents in dataset share the exact same fields)
# ==============================================================================

class SROIEInvoice(StrictExtractionModel):
    """ICDAR 2019 SROIE scanned receipt schema."""
    company: str | None = Field(default=None, description="Name of company or vendor")
    date: str | None = Field(default=None, description="Invoice or receipt date")
    address: str | None = Field(default=None, description="Store address")
    total: str | None = Field(default=None, description="Total amount paid")


class NigeriaMedicalRecord(StrictExtractionModel):
    """Nigeria medical records clinical card schema."""
    name: str | None = Field(default=None, description="Patient name or initials")
    age: str | None = Field(default=None, description="Patient age with units if present")
    sex: str | None = Field(default=None, description="Patient sex / gender")


class IDRBTCheque(StrictExtractionModel):
    """IDRBT Cheque dataset schema."""
    payee_name: str | None = Field(default=None, description="Payee name")
    amount_in_words: str | None = Field(default=None, description="Amount written in words")
    amount: str | None = Field(default=None, description="Amount written in numbers")
    year: str | None = Field(default=None, description="Cheque year")
    account_no: str | None = Field(default=None, description="Bank account number")


class IAMHandwriting(StrictExtractionModel):
    """IAM handwriting transcription schema."""
    text: str | None = Field(default=None, description="Transcription of the handwritten line/text")


FIXED_SCHEMAS: dict[str, type[StrictExtractionModel]] = {
    "SROIE_Invoice Dataset": SROIEInvoice,
    "Nigeria_medical_records": NigeriaMedicalRecord,
    "IDRBT_Cheque_Dataset": IDRBTCheque,
    "IAM handwriting finevision": IAMHandwriting,
}


# ==============================================================================
# Tier 2 & 3: Grouped Template Schemas & Dynamic Template Model Generator
# ==============================================================================

def sanitize_field_name(name: str) -> str:
    """Sanitize a ground truth key into a valid Python identifier for Pydantic."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    cleaned = re.sub(r"^_+|_+$", "", cleaned)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def create_template_model(model_name: str, fields: list[str]) -> type[BaseModel]:
    """Dynamically construct a strict Pydantic model from a list of field names.
    
    Uses Pydantic Field(alias=...) so that the JSON schema uses the exact ground truth
    string while internal Python field access is sanitized.
    """
    field_definitions: dict[str, Any] = {}
    used_identifiers: set[str] = set()

    for field in fields:
        py_name = sanitize_field_name(field)
        if py_name in used_identifiers:
            idx = 1
            while f"{py_name}_{idx}" in used_identifiers:
                idx += 1
            py_name = f"{py_name}_{idx}"
        used_identifiers.add(py_name)
        
        # Define field as nullable string with alias matching ground truth label
        field_definitions[py_name] = (
            str | None,
            Field(default=None, alias=field, description=f"Extracted value for {field}"),
        )

    return create_model(
        model_name,
        __base__=StrictExtractionModel,
        **field_definitions,
    )


# Pre-defined recurring templates for funds_handwritten_dataset
FaxCoverTemplate = create_template_model("FaxCoverTemplate", [
    "DATE", "CLIENT NO.", "MESSAGE TO", "COMPANY", "FAX NUMBER", "PHONE", "FROM", "PHONE (FROM)"
])

ProjectRecordTemplate = create_template_model("ProjectRecordTemplate", [
    "PROJECT CODE", "DATE", "PROJECT NAME", "PROJECT LEADER", "WORK REQUESTED BY", "PROJECT OBJECTIVE", "OTHER PERSONNEL ASSIGNED"
])

AdvertisingCouponTemplate = create_template_model("AdvertisingCouponTemplate", [
    "FROM", "TO", "CC", "BRAND(S) APPLICABLE", "MEDIA TYPE", "MEDIA NAME", "ISSUE FREQUENCY/YEAR",
    "SPACE/COLOR", "COUPON ISSUE DATE", "COUPON EXPIRATION DATE", "CIRCULATION", "GEOGRAPHICAL AREA(S)",
    "COUPON VALUE", "PACK AND/OR CARTON", "*ADVERTISING CREATIVE TITLE", "DATE INITIATED"
])


def to_vllm_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate an OpenAI / vLLM strict-compliant JSON schema from a Pydantic model.
    
    Enforces:
    - type: object
    - additionalProperties: false
    - required: list of all properties (using field aliases if defined)
    - Removes titles, defaults, and extraneous keys that can confuse CFG compilers.
    """
    schema = model.model_json_schema(by_alias=True)
    schema.pop("title", None)
    schema.pop("description", None)
    schema["type"] = "object"
    schema["additionalProperties"] = False

    properties = schema.get("properties", {})
    for prop_name, prop_spec in properties.items():
        prop_spec.pop("title", None)
        # Ensure anyOf or type allows null
        if "anyOf" in prop_spec:
            has_null = any(item.get("type") == "null" for item in prop_spec["anyOf"])
            if not has_null:
                prop_spec["anyOf"].append({"type": "null"})
        elif "type" in prop_spec and prop_spec["type"] != "null":
            prop_spec["anyOf"] = [{"type": prop_spec.pop("type")}, {"type": "null"}]

    schema["required"] = list(properties.keys())
    return schema


def get_schema_for_example(
    dataset: str,
    truth_fields: list[str] | tuple[str, ...],
    mode: str = "template",
) -> tuple[type[BaseModel] | None, dict[str, Any] | None, list[str]]:
    """Resolve the appropriate Pydantic model and vLLM response_format for an example.
    
    Args:
        dataset: Name of the dataset collection.
        truth_fields: List of ground-truth field names for this document.
        mode: "template" for constrained decoding with the document's schema,
              "open" for generic JSON CFG (response_format: json_object).
              
    Returns:
        tuple of (pydantic_model, vllm_response_format, field_names)
    """
    if mode == "open":
        return None, {"type": "json_object"}, list(truth_fields)

    # 1. Fixed Schema Datasets
    if dataset in FIXED_SCHEMAS:
        model = FIXED_SCHEMAS[dataset]
        schema = to_vllm_json_schema(model)
        fields = list(schema.get("properties", {}).keys())
        response_fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": f"{dataset.lower().replace(' ', '_')}_schema",
                "strict": True,
                "schema": schema,
            },
        }
        return model, response_fmt, fields

    # 2. Grouped / Template Datasets (funds_handwritten, Forms600)
    # If ground-truth fields are available, synthesize a strict template model
    clean_name = f"{re.sub(r'[^a-zA-Z0-9]+', '', dataset)}Template"
    model = create_template_model(clean_name, list(truth_fields))
    schema = to_vllm_json_schema(model)
    fields = list(truth_fields)
    response_fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": f"{clean_name.lower()}_schema",
            "strict": True,
            "schema": schema,
        },
    }
    return model, response_fmt, fields

