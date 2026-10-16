#!/usr/bin/env python3
"""Benchmark image-to-JSON extraction models served by an OpenAI-compatible vLLM API."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from metrics import evaluate_prediction, norm_robust, norm_strict
from schemas import (
    FIXED_SCHEMAS,
    get_schema_for_example,
)


@dataclass
class Example:
    dataset: str
    image: Path
    truth: dict[str, str]


class ModelResponseError(ValueError):
    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


SYSTEM_PROMPT = (
    "You are a careful document-digitization assistant. You help users "
    "transform paperwork in image form into faithful digital records. Read "
    "only what is visible in the document. Preserve names, labels, spelling, "
    "capitalization, punctuation, dates, numbers, and handwriting as closely "
    "as possible. Do not guess, complete, normalize, or invent values. "
    "Return only valid JSON because your output is consumed by an automated "
    "benchmark. Use null when a requested value is absent or unreadable."
)


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, child_prefix))
        return result
    if isinstance(value, list):
        # Format list elements cleanly
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                # Represent dictionary in list cleanly as key-value pairs
                flat_item = ", ".join(f"{k}: {v}" for k, v in item.items() if v)
                if flat_item:
                    items.append(flat_item)
            else:
                items.append(str(item))
        return {prefix: " | ".join(items)}
    return {prefix: "" if value is None else str(value)}


def load_json(path: Path) -> dict[str, str]:
    return flatten(json.loads(path.read_text(encoding="utf-8")))


def pair_json_images(
    root: Path, image_dir: str, truth_dir: str, extensions: tuple[str, ...]
) -> tuple[list[Example], list[str]]:
    def image_key(path: Path) -> str:
        return re.sub(r"_HTR(?:_p\d+)?$", "", path.stem, flags=re.IGNORECASE)

    image_path = root / image_dir
    truth_path = root / truth_dir
    if not image_path.is_dir() or not truth_path.is_dir():
        return [], [f"{root.name}: expected {image_dir}/ and {truth_dir}/ directories"]
    images = {
        image_key(p): p
        for p in image_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    }
    truths = {p.stem: p for p in truth_path.glob("*.json") if p.is_file()}
    warnings: list[str] = []
    examples: list[Example] = []
    for stem in sorted(images.keys() & truths.keys()):
        examples.append(Example(root.name, images[stem], load_json(truths[stem])))
    for stem in sorted(truths.keys() - images.keys()):
        warnings.append(f"{root.name}: missing image for {truths[stem].relative_to(root)}")
    for stem in sorted(images.keys() - truths.keys()):
        warnings.append(f"{root.name}: missing truth for {images[stem].relative_to(root)}")
    return examples, warnings


def discover(data_dir: Path) -> tuple[list[Example], list[str]]:
    examples: list[Example] = []
    warnings: list[str] = []
    forms = data_dir / "Forms600"
    sroie = data_dir / "SROIE_Invoice Dataset"
    nigeria = data_dir / "Nigeria_medical_records"
    fax = data_dir / "funds_handwritten_dataset"
    cheques = data_dir / "IDRBT_Cheque_Dataset"

    examples_, warnings_ = pair_json_images(forms, "Images", "Ground_truth", (".png", ".jpg", ".jpeg"))
    examples += examples_; warnings += warnings_
    examples_, warnings_ = pair_json_images(sroie, "Images", "Ground_truth", (".png", ".jpg", ".jpeg"))
    examples += examples_; warnings += warnings_
    examples_, warnings_ = pair_json_images(nigeria, "Images", "Truth", (".png", ".jpg", ".jpeg"))
    examples += examples_; warnings += warnings_
    examples_, warnings_ = pair_json_images(fax, "images", "ground_truth", (".png", ".jpg", ".jpeg"))
    examples += examples_; warnings += warnings_
    examples_, warnings_ = pair_json_images(cheques, "Images", "Truth", (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    examples += examples_; warnings += warnings_

    csv_path = data_dir / "ground_truth_40.csv"
    image_dir = data_dir / "images"
    if csv_path.exists() and image_dir.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image = image_dir / row["filename"]
                truth = {"text": row["ground_truth"]}
                if image.exists():
                    examples.append(Example("IAM handwriting finevision", image, truth))
                else:
                    warnings.append(f"IAM handwriting finevision: missing image {image}")
    return examples, warnings


def image_payload(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def chat_completions_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def parse_json_content(content: str) -> dict[str, Any]:
    candidates = [content.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(content[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("model response did not contain a JSON object")


def request_prediction(
    base_url: str,
    model: str,
    image: Path,
    dataset: str,
    truth_fields: list[str],
    schema_mode: str,
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, str], str, dict[str, int]]:
    pydantic_model, response_fmt, fields = get_schema_for_example(
        dataset, truth_fields, mode=schema_mode
    )

    if response_fmt.get("type") == "json_schema":
        prompt = (
            "Extract the values of the following predefined fields from this "
            "document image. Use exactly these field names. Return a valid JSON object. "
            "Use null when a field is not present or unreadable.\nFields:\n"
            + json.dumps(fields, ensure_ascii=False)
        )
    else:
        prompt = (
            "Extract all meaningful labeled fields and their values from this "
            "document image. Do not assume a predefined schema. Preserve the "
            "field labels as JSON keys. Return a valid JSON object of key-value pairs; "
            "use null for a field that is present but unreadable."
        )

    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_payload(image)}},
                ],
            },
        ],
        "response_format": response_fmt,
    }

    # Pass guided_json for vLLM versions that require it via extra_body
    if response_fmt.get("type") == "json_schema":
        body["extra_body"] = {
            "guided_json": response_fmt["json_schema"]["schema"]
        }

    request = Request(
        chat_completions_endpoint(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raw_response = str(content)
    usage = payload.get("usage", {})

    try:
        parsed = parse_json_content(raw_response)
        # If a strict Pydantic model is available, validate against it
        if pydantic_model is not None:
            try:
                validated = pydantic_model.model_validate(parsed)
                parsed = validated.model_dump(by_alias=True)
            except Exception:
                # If validation has minor key discrepancies, preserve parsed dict
                pass
        return flatten(parsed), raw_response, usage
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        finish_reason = payload.get("choices", [{}])[0].get("finish_reason", "unknown")
        raise ModelResponseError(
            f"model response was not valid JSON (finish_reason={finish_reason}): {exc}",
            raw_response,
        ) from exc


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "image",
        "field",
        "ground_truth",
        "prediction",
        "strict_match",
        "robust_match",
        "ned_similarity",
        "manual_status",
        "manual_notes",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            details = row.get("field_details", {})
            for field, d in details.items():
                writer.writerow({
                    "dataset": row["dataset"],
                    "image": row["image"],
                    "field": field,
                    "ground_truth": d["truth"],
                    "prediction": d["prediction"],
                    "strict_match": d["strict_match"],
                    "robust_match": d["robust_match"],
                    "ned_similarity": d["ned_similarity"],
                    "manual_status": "",
                    "manual_notes": "",
                    "error": row["error"],
                })


def evaluate_example(
    example: Example,
    base_url: str,
    model: str,
    schema_mode: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    prediction: dict[str, str] = {}
    error = ""
    raw = ""
    usage: dict[str, int] = {}
    truth_fields = list(example.truth.keys())

    try:
        prediction, raw, usage = request_prediction(
            base_url,
            model,
            example.image,
            example.dataset,
            truth_fields,
            schema_mode,
            max_tokens,
            timeout,
        )
    except ModelResponseError as exc:
        raw = exc.raw_response
        error = f"{type(exc).__name__}: {exc}"
    except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    metrics = evaluate_prediction(example.truth, prediction)
    latency = round(time.perf_counter() - started, 3)

    return {
        "dataset": example.dataset,
        "image": str(example.image),
        "truth": example.truth,
        "prediction": prediction,
        "field_count": metrics["truth_count"],
        "pred_count": metrics["pred_count"],
        "strict_correct": metrics["strict_correct"],
        "robust_correct": metrics["robust_correct"],
        "form_correct_strict": bool(not error and metrics["form_correct_strict"]),
        "form_correct_robust": bool(not error and metrics["form_correct_robust"]),
        "precision_strict": metrics["precision_strict"],
        "recall_strict": metrics["recall_strict"],
        "f1_strict": metrics["f1_strict"],
        "precision_robust": metrics["precision_robust"],
        "recall_robust": metrics["recall_robust"],
        "f1_robust": metrics["f1_robust"],
        "mean_ned": metrics["mean_ned"],
        "field_details": metrics["field_details"],
        "latency_seconds": latency,
        "tokens_prompt": usage.get("prompt_tokens", 0),
        "tokens_completion": usage.get("completion_tokens", 0),
        "error": error,
        "raw_response": raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/unpacked"))
    parser.add_argument(
        "--endpoint",
        "--base-url",
        dest="base_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL, normally ending in /v1",
    )
    parser.add_argument("--model", required=True, help="Model name exposed by vLLM")
    parser.add_argument(
        "--schema-mode",
        choices=["template", "open"],
        default="template",
        help="Constrained decoding mode: 'template' (Pydantic schema constraints) or 'open' (generic json_object)",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help="Per-field CSV for manual review (default: beside --output with .review.csv suffix)",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens per request",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (vLLM continuous batching)",
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    examples, warnings = discover(args.data_dir)
    if args.max_samples:
        examples = examples[: args.max_samples]
    if not examples:
        print("No paired image/truth examples found.", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    rows: list[dict[str, Any] | None] = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                evaluate_example,
                example,
                args.base_url,
                args.model,
                args.schema_mode,
                args.max_tokens,
                args.timeout,
            ): index
            for index, example in enumerate(examples)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            row = future.result()
            rows[index] = row
            status_str = f"strict={row['strict_correct']}/{row['field_count']} robust={row['robust_correct']}/{row['field_count']} F1={row['f1_robust']}"
            if row["error"]:
                status_str += f" ERROR: {row['error']}"
            print(f"[{completed}/{len(examples)}] {row['dataset']}: {status_str}")

    completed_rows = [row for row in rows if row is not None]

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in completed_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summary: dict[str, Any] = {}
    for dataset, dataset_rows in by_dataset.items():
        n = len(dataset_rows)
        total_truth_fields = sum(r["field_count"] for r in dataset_rows)
        total_pred_fields = sum(r["pred_count"] for r in dataset_rows)
        strict_correct = sum(r["strict_correct"] for r in dataset_rows)
        robust_correct = sum(r["robust_correct"] for r in dataset_rows)

        summary[dataset] = {
            "forms": n,
            "forms_correct_strict": sum(r["form_correct_strict"] for r in dataset_rows),
            "forms_correct_robust": sum(r["form_correct_robust"] for r in dataset_rows),
            "form_accuracy_strict": sum(r["form_correct_strict"] for r in dataset_rows) / n,
            "form_accuracy_robust": sum(r["form_correct_robust"] for r in dataset_rows) / n,
            "mean_field_accuracy_strict": sum(r["recall_strict"] for r in dataset_rows) / n,
            "mean_field_accuracy_robust": sum(r["recall_robust"] for r in dataset_rows) / n,
            "micro_field_accuracy_strict": strict_correct / total_truth_fields if total_truth_fields else 0.0,
            "micro_field_accuracy_robust": robust_correct / total_truth_fields if total_truth_fields else 0.0,
            "precision_strict": strict_correct / total_pred_fields if total_pred_fields else 0.0,
            "precision_robust": robust_correct / total_pred_fields if total_pred_fields else 0.0,
            "f1_strict": sum(r["f1_strict"] for r in dataset_rows) / n,
            "f1_robust": sum(r["f1_robust"] for r in dataset_rows) / n,
            "mean_ned": sum(r["mean_ned"] for r in dataset_rows) / n,
            "avg_latency": sum(r["latency_seconds"] for r in dataset_rows) / n,
            "errors": sum(bool(r["error"]) for r in dataset_rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    review_output = args.review_output or args.output.with_suffix(".review.csv")
    write_review_csv(review_output, completed_rows)

    result = {
        "config": {
            "base_url": args.base_url,
            "request_endpoint": chat_completions_endpoint(args.base_url),
            "model": args.model,
            "schema_mode": args.schema_mode,
            "concurrency": args.concurrency,
        },
        "summary": summary,
        "examples": completed_rows,
        "warnings": warnings,
        "manual_review_csv": str(review_output),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Benchmark Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved detailed results to {args.output}")
    print(f"Saved manual-review CSV to {review_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
