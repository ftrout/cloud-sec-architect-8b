# cloud-sec-architect-8b

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-orange)](https://huggingface.co/fmt0816/cloud-sec-architect-8b)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A specialized fine-tuned LLM designed to function as a **Senior Cloud Security Architect**. Built on Meta's Llama 3.1 8B Instruct, this model provides expert-level guidance on multi-cloud security architecture, compliance frameworks, and infrastructure-as-code security.

---

## Key Features

* **Multi-Cloud Expertise:** Deep knowledge of **AWS**, **Azure**, **GCP**, and **Kubernetes** security best practices.
* **Compliance & Standards:** Trained on **NIST 800-53**, **CIS Benchmarks**, **OWASP Top 10**, and **MITRE ATT&CK** for cloud.
* **Identity & IaC Focus:** Specialized modules for **Identity Architecture** (OIDC/SAML) and **Policy-as-Code** (Terraform/OPA).
* **High-Fidelity Data Engine:** Uses a custom `trafilatura` + `MinHash` pipeline to harvest, clean, and deduplicate technical documentation, ensuring zero "marketing fluff" enters the training set.
* **Reasoning-First Model:** Built on **Llama 3.1 8B Instruct**, optimized for complex architectural reasoning and "System 2" thinking.

---

## Quick Start

### Installation

```bash
git clone https://github.com/ftrout/cloud-sec-architect-8b.git
cd cloud-sec-architect-8b
pip install -r requirements.txt
```

### Run the Demo

```bash
# Gradio web interface
python demo.py

# Command-line inference
python scripts/inference.py --interactive
```

### Use from Hugging Face Hub

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(model, "fmt0816/cloud-sec-architect-8b")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
```

---

## Prerequisites

### Hardware Requirements

| Use Case | GPU VRAM | Example GPUs |
|----------|----------|--------------|
| **Training** | ≥24GB | RTX 3090, RTX 4090, A10G, A100 |
| **Inference** | ≥12GB | RTX 3060, RTX 4070, T4 |
| **Disk Space** | ~50GB | For datasets and checkpoints |

### Software Stack

* **OS:** Linux (Ubuntu 22.04 LTS recommended) or WSL2
* **Python:** 3.10+
* **Core Libraries:** `torch`, `transformers`, `peft`, `trl`, `trafilatura`, `datasketch`

---

## The Pipeline

### Step 1: Data Engine (`harvest_data.py`)

Crawls curated professional documentation (AWS, Azure, GCP, Kubernetes, NIST, CIS), extracts main content, filters for technical depth (Grade Level > 8), and removes semantic duplicates.

```bash
python harvest_data.py
```

* **Inputs:** Curated list of high-value domains (AWS, Azure, K8s, CIS, MITRE).
* **Process:** Fetch → Extract Main Text → Quality Filter → Deduplicate → Chunk.
* **Output:** `./data/architect_training_data.jsonl`

### Step 2: Training Lab (`train.py`)

Fine-tunes **Llama 3.1 8B** using **QLoRA** (Quantized Low-Rank Adaptation). Injects a "Senior Architect" persona and trains the model to provide precise, compliant technical guidance.

```bash
python train.py
```

* **Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Technique:** 4-bit Quantization + LoRA (Rank 32)
* **Output:** `./cloud-sec-architect-8b` (The trained adapter weights)
* **Training Time:** ~2-4 hours on a single RTX 3090

### Step 3: Evaluation (`evaluate.py`)

Validates the model against a "Golden Set" of complex architectural questions to ensure reasoning capability.

```bash
python evaluate.py
```

---

## Project Structure

```text
cloud-sec-architect-8b/
├── config/                     # Configuration files
│   └── training_config.yaml
├── data/                       # Generated datasets
│   └── architect_training_data.jsonl
├── scripts/                    # Utility scripts
│   ├── inference.py            # CLI inference
│   └── upload_to_hub.py        # HuggingFace upload
├── cloud-sec-architect-8b/     # Saved LoRA Adapters
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── results/                    # Training checkpoints
├── tests/                      # Unit tests
│   ├── test_config.py
│   └── test_harvester.py
├── harvest_data.py             # Data Engine
├── train.py                    # Training Script
├── evaluate.py                 # Evaluation Script
├── demo.py                     # Gradio Web Interface
├── MODEL_CARD.md               # HuggingFace Model Card
├── DATASET_CARD.md             # Dataset Documentation
├── CODE_OF_CONDUCT.md          # Community Guidelines
├── FAQ.md                      # Frequently Asked Questions
├── SECURITY.md                 # Security Policy
├── requirements.txt            # Dependencies
├── pyproject.toml              # Package configuration
└── README.md                   # Documentation
```

---

## Model Architecture

| Component | Specification |
|-----------|---------------|
| **Base Model** | Llama 3.1 8B Instruct |
| **Fine-tuning Method** | QLoRA (4-bit NF4) |
| **LoRA Rank (r)** | 32 |
| **LoRA Alpha** | 64 |
| **Target Modules** | `all-linear` (Self-Attention & MLP layers) |
| **Context Window** | 2048 tokens (training) |

---

## Uploading to Hugging Face

After training, upload your model to Hugging Face Hub:

```bash
# Upload LoRA adapter (recommended - smaller size)
python scripts/upload_to_hub.py --repo-id fmt0816/cloud-sec-architect-8b

# Upload merged full model (larger but easier to use)
python scripts/upload_to_hub.py --repo-id fmt0816/cloud-sec-architect-8b-merged --merge
```

---

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
mypy .
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## Disclaimer

This tool is for educational and research purposes. While trained on authoritative sources, AI models can hallucinate. Always verify architectural decisions against official vendor documentation and your organization's compliance requirements before implementation.

---

## Citation

```bibtex
@misc{cloud-sec-architect-8b,
  author = {Trout, Frank},
  title = {cloud-sec-architect-8b: A Fine-tuned LLM for Cloud Security Architecture},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/fmt0816/cloud-sec-architect-8b}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
