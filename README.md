# Forms benchmarking

The archives in `datasets/` have been unpacked into `datasets/unpacked/`. The
benchmark covers every image with a matching ground-truth file:

| Dataset | Label format | Expected task |
|---|---|---|
| Forms600 | Nested JSON | Generic form key/value extraction |
| SROIE Invoice Dataset | Flat JSON | Invoice field extraction |
| Nigeria medical records | Flat JSON | Medical-record field extraction |
| funds_handwritten_dataset | Flat JSON | Handwritten fax/form extraction |
| IDRBT Cheque Dataset | Flat JSON | Cheque field extraction |
| IAM handwriting finevision | CSV | Single-field handwriting transcription |

Run a vLLM server with its OpenAI-compatible multimodal Chat Completions API,
then run:

```bash
uv run python benchmark.py --model nuextract3
```

The endpoint and model are configurable:

```bash
uv run python benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model nuextract3 \
  --output results/nuextract3.json
```

The default output budget is 4096 tokens. Increase it for documents with many
long values:

```bash
uv run python benchmark.py --model nuextract3 --max-tokens 8192
```

For throughput, use independent concurrent requests. Each request still sends
exactly one image; vLLM's scheduler performs continuous/in-flight batching
across those requests:

```bash
uv run python benchmark.py \
  --model nuextract3 \
  --concurrency 4
```

Start with a small value such as 2 or 4 and increase only while the server has
available GPU memory and request latency remains acceptable. This is different
from placing multiple images in one prompt. The benchmark preserves the
original dataset order in the result JSON even though completion logs may
arrive out of order.

`--endpoint` remains accepted as an alias for `--base-url`. The value is the
OpenAI-compatible base URL, not the full operation URL. The default is
`http://localhost:8000/v1`; the script derives
`/v1/chat/completions` internally. A full URL ending in
`/chat/completions` is also accepted.

The script uses vLLM's Chat Completions API with constrained decoding:
* In `--schema-mode template` (default), fixed-schema datasets (SROIE, Nigeria medical
  records, IDRBT cheques, IAM handwriting) and grouped-schema datasets (funds handwritten,
  Forms600) use strict Pydantic v2 models. The schema is compiled into an OpenAI/vLLM-compliant
  JSON schema (`additionalProperties: false`, all fields in `required` array) and passed
  via `response_format={"type": "json_schema", ...}` as well as `extra_body={"guided_json": ...}`.
* In `--schema-mode open`, arbitrary forms use vLLM's CFG-constrained generic JSON mode
  `response_format={"type": "json_object"}`, ensuring valid JSON syntax without predefined keys.

Evaluation metrics include:
* **Form accuracy (Strict & Robust)**: proportion of forms for which every ground-truth field
  is recovered. Strict requires exact normalized string match; Robust applies domain-aware
  normalization (currency symbols, numeric trailing zeros, commas, date representations, abbreviations).
* **Mean field accuracy (Strict & Robust)**: per-form recall of ground-truth fields, averaged across forms.
* **Micro field accuracy (Strict & Robust)**: total correct fields divided by total ground-truth fields.
* **Field Precision, Recall & F1**: evaluates true extraction completeness and penalizes hallucinations.
* **Normalized Edit Distance (NED)**: character-level similarity metric ($1 - \text{Levenshtein} / \max$)
  especially vital for handwriting transcription and noisy scans.

Run comparative analysis and generate LaTeX tables with:
```bash
uv run python analyze_results.py \
  --nuextract results/nuextract3.json \
  --granite results/granite-vision.json \
  --update-report progress_report.tex
```

Detailed per-example predictions, raw responses, latency, errors, warnings, and
dataset summaries are written to the JSON file passed via `--output`. The
unpacked files are intentionally retained so runs are reproducible.
If a model response is malformed or truncated, the raw response is still
retained in the JSON result and the error records the API `finish_reason` when
available. This supports manual diagnosis instead of silently discarding the
response.

The benchmark also writes a per-field manual-review CSV beside that JSON by
default. For example, `results/nuextract3.json` produces
`results/nuextract3.review.csv`. Each row contains the dataset, image path,
field name, ground truth, model prediction, automatic exact-match result, and
blank `manual_status`/`manual_notes` columns for a reviewer to fill in. Extra
predicted fields are included as rows too, making omissions and hallucinated
fields visible. Override the path with `--review-output`.

This review artifact is important because automatic exact match is not a
complete correctness judgment: dates, abbreviations, punctuation, handwriting,
and semantically equivalent values may need human adjudication. The original
model response is also retained in the JSON result under `raw_response`.

## Relevant benchmarks and prior art

* **FUNSD**: form understanding in noisy scanned documents; its entity and
  relation annotations motivate evaluating field recovery separately from
  whole-form exact match. [Paper](https://arxiv.org/abs/1905.13538)
* **SROIE**: ICDAR 2019 scanned receipt OCR and information extraction
  competition; the local SROIE subset uses the familiar company/date/address/
  total fields. [Paper](https://arxiv.org/abs/2103.10213)
* **CORD**: receipt understanding benchmark, useful for later expanding beyond
  exact flat key/value labels. [Dataset](https://github.com/clovaai/cord)
* **Donut**: OCR-free document understanding transformer, a relevant
  image-to-structured-output baseline. [Paper](https://arxiv.org/abs/2111.15664)
* **DocVQA**: document visual question answering benchmark; it is relevant if
  the extraction interface is changed from all-fields JSON to field questions.
  [Task site](https://www.docvqa.org/)

These references use different annotation schemes and often report entity
precision/recall/F1 or normalized edit distance rather than whole-form exact
match. The two requested metrics are therefore reported as the primary local
metrics, while micro field accuracy and raw per-field results make comparisons
with those conventions possible. Exact match is intentionally strict and
should be supplemented with normalization or field-level F1 for production
decisions.

## Initial implementation report

### Scope completed

* Re-extracted all six current ZIP archives into `datasets/unpacked/`.
* Added automatic discovery and pairing for all six layouts, including TIFF
  cheque images.
* Added an OpenAI-compatible vLLM Chat Completions client with configurable
  base URL, model, timeout, request pacing, and JSON-constrained output.
* Added fenced/free-form JSON response recovery, nested-label flattening,
  per-example error capture, latency, and aggregate summaries.
* Added per-field CSV logging for manual verification, including extra model
  fields and reviewer status/notes columns.
* Added schema-guided versus open-schema dataset handling.
* Added configurable concurrent one-image requests to use vLLM continuous
  batching for throughput without combining images in a prompt.
* Added a document-digitization assistant system prompt focused on faithful
  transcription and non-hallucination.

### Initial inventory

The extracted collections contain 25 Forms600 images, 25 SROIE images, 22
Nigeria medical-record images, 26 handwritten fax images from
`funds_handwritten_dataset`, 25 IDRBT cheque images, and 40 IAM handwriting
images (subject to the pairing warnings printed by the script).
The script is the source of truth for the final runnable count because it
excludes only unpaired files and records those exclusions.

### Recommended next experiment

Run the same fixed sample order and prompt for each model/endpoint, preserve
the JSON result files, and compare form accuracy, mean field accuracy, micro
field accuracy, error rate, and latency by dataset. Before using scores as a
release gate, add dataset-specific normalization (dates, currency, phone
numbers), field-level precision/recall/F1, and a manually reviewed sample of
JSON parses. The current strict comparison is useful as a reproducible
baseline but can penalize semantically equivalent formatting.
