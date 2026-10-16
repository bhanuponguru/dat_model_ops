# ModelOps for Health Insurance AI Agents: Form Digitization Benchmark

**Authors**: Ponguru Bhanu (CS23BTECH11046) & Krapa Sandeep (CS23BTECH11027)  
**Affiliation**: Department of Computer Science and Engineering, IIT Hyderabad  
**Domain Focus**: ModelOps, Privacy-Preserving On-Premises Vision-Language Models, Constrained Decoding

---

## 1. Overview & Problem Formulation

Autonomous insurance agents automate claims adjudication, pre-authorizations, coverage checks, and fraud triage. A foundational component in these pipelines is **form digitization**: converting heterogeneous physical and scanned documents into structured, validated JSON entities for downstream rule engines.

This repository implements a reproducible benchmarking suite and ModelOps evaluation framework designed around two core operational constraints:

1. **Data Privacy Mandates (PHI/HIPAA)**: Medical claims and banking records contain Protected Health Information (PHI). Data protection regulations restrict transmitting raw document images to third-party public cloud APIs, requiring the document intake pipeline to run strictly on-premises.
2. **Compute Constraints**: Regional insurance administrators, hospitals, and third-party administrators (TPAs) operate under restricted compute budgets, requiring small vision-language models ($\le 4\text{B}$ parameters) executable on local commodity GPUs.
3. **Structured Constrained Decoding**: Unconstrained autoregressive generation produces syntax errors, invented keys, and formatting drift. Candidate models are evaluated under token-level logit masking via vLLM and Pydantic Context-Free Grammars (CFGs).

---

## 2. Benchmark Architecture & Candidate Models

The pipeline serves models locally via vLLM's OpenAI-compatible Chat Completions API with grammar-guided logit masking (`response_format={"type": "json_schema", ...}`).

### Candidate Models Evaluated
- **NuExtract 3** (~3.8B parameters): Vision-language model fine-tuned specifically for structured information extraction.
- **IBM Granite Vision 3.1-2B** (~2B parameters): Compact generalist multimodal foundation model.

---

## 3. Dataset Inventory

The benchmark encompasses **163 paired evaluation documents** across six collections covering real-world insurance intake formats:

| Dataset Collection | Documents | Target Fields | Document Archetype |
| :--- | :---: | :--- | :--- |
| **Forms-600** | 25 | 11–19 demographic, academic, and banking fields | Multi-column administrative applications |
| **SROIE Invoices** | 25 | `company`, `date`, `address`, `total` | Commercial & pharmacy receipts |
| **Nigerian Medical Records** | 22 | `name`, `age`, `sex` | Handwritten clinical encounter cards |
| **Funds Administrative Forms** | 26 | 8–18 structured header and memo fields | Administrative faxes and cover sheets |
| **IDRBT Banking Cheques** | 25 | `payee_name`, `amount_legal`, `amount`, `year`, `account_no` | Financial verification cheques |
| **IAM Cursive Handwriting** | 40 | Full-line handwriting transcription | Baseline visual OCR |
| **Total Benchmark Suite** | **163** | — | — |

Datasets are stored in `datasets/unpacked/`.

---

## 4. Evaluation Protocol & Multi-Perspective Metrics

Model predictions are evaluated across multiple complementary dimensions:

- **Strict Whole-Form Accuracy**: Fraction of documents where every target field matches ground truth exactly after lowercasing and whitespace collapsing.
- **Domain-Aware Robust Whole-Form Accuracy**: Evaluated after canonicalizing domain-specific surface differences:
  - *Dates*: Canonicalized to ISO `YYYY-MM-DD` (e.g., `15/01/18` $\rightarrow$ `2018-01-15`).
  - *Currencies & Numbers*: Stripping currency prefixes (`RM`, `$`), decimal padding, and Indian comma formatting (`88,00,000` $\rightarrow$ `8800000`).
  - *Categorical & Units*: Standardizing sex codes (`M`/`F` $\rightarrow$ `male`/`female`) and stripping age unit suffixes (`41 years` $\rightarrow$ `41`).
- **Mean Field Accuracy (Strict & Robust)**: Per-document field accuracy averaged across the collection.
- **Micro-Averaged Precision, Recall & F1-Score**: Overall entity coverage penalizing missing, extra, and misread fields.
- **Normalized Edit Distance (NED)**: Character-level transcription accuracy:
  $$\text{NED}(y, \hat{y}) = 1 - \frac{\text{Levenshtein}(y, \hat{y})}{\max(|y|, |\hat{y}|)}$$
- **Operational Latency & Decoding Stability**: End-to-end request latency and termination status tracking (`stop` vs `length` token exhaustion).

---

## 5. Aggregate Benchmark Results

Results across all 163 documents under Pydantic logit masking:

| Dataset Collection | Docs | Whole Strict (Nu / Granite) | Whole Robust (Nu / Granite) | Field Strict (Nu / Granite) | Field Robust (Nu / Granite) | Robust F1 (Nu / Granite) | Mean NED (Nu / Granite) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forms-600 Admissions** | 25 | 4.0% / 4.0% | 4.0% / 4.0% | 65.3% / 63.0% | 70.0% / 65.6% | 0.700 / 0.656 | 83.5% / 80.2% |
| **SROIE Invoices** | 25 | 36.0% / 0.0% | 40.0% / 20.0% | 76.0% / 36.0% | 79.0% / 44.0% | 0.790 / 0.440 | 93.2% / 51.3% |
| **Nigerian Medical** | 22 | 9.1% / 4.5% | 31.8% / 9.1% | 42.4% / 42.4% | 72.7% / 59.1% | 0.727 / 0.591 | 69.3% / 69.2% |
| **Funds Administrative** | 26 | 46.2% / 38.5% | 46.2% / 38.5% | 89.6% / 79.3% | 90.7% / 79.8% | 0.907 / 0.798 | 97.9% / 86.3% |
| **IDRBT Cheques** | 25 | 24.0% / 0.0% | 28.0% / 0.0% | 75.2% / 36.0% | 78.4% / 40.0% | 0.784 / 0.400 | 95.6% / 54.3% |
| **IAM Handwriting** | 40 | 37.5% / 37.5% | 40.0% / 40.0% | 37.5% / 37.5\% | 40.0% / 40.0% | 0.400 / 0.400 | 98.2% / 94.5% |
| **Overall / Mean** | **163** | **26.1% / 7.4%** | **31.7% / 18.6%** | **64.3% / 49.0%** | **71.8% / 54.8%** | **0.718 / 0.548** | **89.6% / 72.6%** |

### Operational Profiles
- **NuExtract 3**: Mean latency of 2.58s per document. Achieved 100% normal termination (`finish_reason='stop'`) with zero token exhaustion crashes.
- **IBM Granite Vision**: Achieved fast inference on single cursive lines (0.92s on IAM handwriting). However, it suffered 19 token-exhaustion loop terminations (`finish_reason='length'`) across SROIE receipts (9) and IDRBT cheques (10), increasing average cheque latency to 25.5s.

---

## 6. Qualitative Findings & Diagnosed Failure Modes

Evaluating both models head-to-head on the identical document across four form archetypes revealed four recurring failure modes:

1. **1D Spatial Coordinate Disconnect**: On multi-column forms (Forms-600 Form 19), large horizontal whitespace margins separating labels and values lead to field dropouts (NuExtract: 26.3% accuracy, Granite: 47.4%). Autoregressive token decoders lack explicit 2D bounding-box coordinates.
2. **CFG Whitespace Repetition Loops**: Standard JSON Context-Free Grammars permit arbitrary whitespace (`\s*`). On ambiguous documents (SROIE Receipt 3, Cheque 309097, Form 7), Granite Vision's low-confidence policy falls into infinite space/tab loops until hitting the 4096-token ceiling.
3. **Physical Scan Degradation**: Low-contrast ballpoint text (AMR_057) caused Granite to emit null strings, whereas NuExtract extracted all fields. Reverse-side carbon ink bleed (AMR_058) caused character misreadings (`P. J` $\rightarrow$ `P. I`).
4. **Clinical Normalization Discrepancy**: On medical card AMR_001, both models scored 0% strict accuracy despite reading `41` and `F` correctly because reference labels contained `41 years` and `Female`. Domain normalization yielded 100% accuracy, reflecting a +30.3% gap across the clinical collection.

---

## 7. ModelOps Next Steps: Fine-Tuning & Deployment Roadmap

The benchmark results define the subsequent development phases:

1. **Supervised Fine-Tuning (SFT) with 2D Coordinate Grounding**:
   - Train on an expanded domain dataset (claims, bills, cards, cheques) with cross-entropy loss:
     $$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, X_{\text{image}}, S_{\text{schema}})$$
   - Incorporate 2D bounding-box spatial coordinates and synthetic scan noise (blur, bleed, contrast downsampling) to resolve table and multi-column field dropouts.
2. **Preference Alignment (RLHF / DPO)**:
   - Optimize model weights using a multi-objective reward function evaluated against reference annotations $y^*$:
     $$R(y, y^*) = \alpha_1 \cdot \text{NED}(y, y^*) + \alpha_2 \cdot F_1(y, y^*) - \alpha_3 \cdot \mathcal{P}_{\text{rep}}(y) - \alpha_4 \cdot \mathcal{P}_{\text{format}}(y)$$
   - $\mathcal{P}_{\text{rep}}$ penalizes repetitive whitespace and token loops to eliminate CFG loop traps.
   - $\mathcal{P}_{\text{format}}$ penalizes non-canonical date formats and unnormalized units.
3. **On-Premises Deployment**:
   - Serve the fine-tuned model checkpoint locally via vLLM with Pydantic logit masking for private, high-throughput intake in insurance agents.

---

## 8. Repository Structure

```
├── benchmark.py                 # Multi-domain benchmarking engine with concurrent vLLM requests
├── schemas.py                   # Pydantic v2 schemas and JSON Schema compilation
├── metrics.py                   # Strict/Robust exact match, F1, NED, latency, and failure metrics
├── analyze_results.py           # Result aggregation, statistical summaries, and LaTeX table generator
├── generate_slides.py           # 16:9 presentation slide deck generator (python-pptx)
├── progress_report.tex          # LaTeX technical progress report source
├── progress_report.pdf          # Compiled 9-page technical report
├── datasets/
│   └── unpacked/               # Extracted images, ground-truth JSONs, and CSV annotations
├── tests/                       # Automated test suite (18 unit and integration tests)
├── pyproject.toml               # Python project configuration and dependency specifications
└── uv.lock                      # Deterministic dependency lockfile
```

---

## 9. Setup & Reproduction Guide

### Prerequisites
- Python 3.10+ (managed via `uv`)
- A running vLLM server with multimodal Chat Completions enabled (`--model ...`)

### Installation
```bash
git clone https://github.com/bhanuponguru/dat_model_ops.git
cd dat_model_ops
uv sync
```

### Running Tests
Execute the 18-test verification suite:
```bash
uv run python -m pytest -v
```

### Running the Benchmark
```bash
# Benchmark NuExtract 3
uv run python benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model nuextract3 \
  --concurrency 4 \
  --output results/nuextract3.json

# Benchmark IBM Granite Vision 3.1-2B
uv run python benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model ibm-granite/granite-vision-3.1-2b-preview \
  --concurrency 4 \
  --output results/granite-vision.json
```

### Analyzing Results & Updating Reports
```bash
uv run python analyze_results.py \
  --nuextract results/nuextract3.json \
  --granite results/granite-vision.json \
  --update-report progress_report.tex
```

### Generating Presentation Slides
```bash
uv run python generate_slides.py -o presentation_slides.pptx
```
