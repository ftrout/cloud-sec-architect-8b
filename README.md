# Cloud-Sec-Architect-AI

`Cloud-Sec-Architect-AI` is a specialized fine-tuning pipeline designed to create an AI model capable of functioning as a **Senior Cloud Security Architect**. Unlike generic LLMs, this model is rigorously trained on high-fidelity, vendor-agnostic documentation, threat frameworks, and infrastructure-as-code (IaC) standards. It excels at designing secure multi-cloud environments, performing gap analysis, and mapping technical implementations to compliance frameworks (NIST, CIS, ISO).

---

## 🚀 Key Features

* **Multi-Cloud Expertise:** Deep knowledge of **AWS**, **Azure**, **GCP**, and **Kubernetes** security best practices.
* **Compliance & Standards:** Trained on **NIST 800-53**, **CIS Benchmarks**, **OWASP Top 10**, and **MITRE ATT&CK** for cloud.
* **Identity & IaC Focus:** Specialized modules for **Identity Architecture** (OIDC/SAML) and **Policy-as-Code** (Terraform/OPA).
* **High-Fidelity Data Engine:** Uses a custom `trafilatura` + `MinHash` pipeline to harvest, clean, and deduplicate technical documentation, ensuring zero "marketing fluff" enters the training set.
* **Reasoning-First Model:** Built on **Llama 3.1 8B Instruct**, optimized for complex architectural reasoning and "System 2" thinking.

---

## 🛠️ Prerequisites

### Hardware Requirements

* **Training:** NVIDIA GPU with **≥24GB VRAM** (RTX 3090, 4090, or A10G) recommended for efficient 4-bit LoRA training.
* **Inference:** NVIDIA GPU with **≥12GB VRAM** (RTX 3060/4070) for 4-bit quantized inference.
* **Disk Space:** ~50GB for datasets and model checkpoints.

### Software Stack

* **OS:** Linux (Ubuntu 22.04 LTS recommended) or WSL2.
* **Python:** 3.10+.
* **Core Libraries:** `torch`, `transformers`, `peft`, `trl`, `trafilatura`, `datasketch`.

---

## 📦 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/Cloud-Sec-Architect-AI.git
cd Cloud-Sec-Architect-AI
```


2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate
```


3. **Install Dependencies**
```bash
pip install -r requirements.txt
```


4. **Hugging Face Login** (Required for Llama 3.1)
```bash
huggingface-cli login
# Enter your HF Token with access to meta-llama/Meta-Llama-3.1-8B-Instruct
```



---

## 🔄 The Pipeline

### Step 1: The Data Engine (`harvest_data.py`)

This script acts as the "Brain" of the operation. It crawls a curated list of professional documentation roots (AWS Prescriptive Guidance, Azure Well-Architected, NIST, etc.), extracts the main content, checks strictly for technical depth (Grade Level > 8), and removes semantic duplicates.

```bash
python harvest_data.py
```

* **Inputs:** Curated list of high-value domains (AWS, Azure, K8s, CIS, MITRE).
* **Process:** Fetch -> Extract Main Text -> Quality Filter -> Deduplicate -> Chunk.
* **Output:** `./data/architect_training_data.jsonl`

### Step 2: The Training Lab (`train.py`)

Fine-tunes **Llama 3.1 8B** using **QLoRA** (Quantized Low-Rank Adaptation). This script injects a "Senior Architect" persona into the system prompt and trains the model to answer user queries with precise, compliant technical guidance.

```bash
python train.py
```

* **Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`.
* **Technique:** 4-bit Quantization + LoRA (Rank 32).
* **Output:** `./cloud-architect-v1` (The trained adapter weights).
* **Training Time:** ~2-4 hours on a single RTX 3090.

### Step 3: The Judge (`evaluate.py`)

Validates the model against a "Golden Set" of complex architectural questions. This ensures the model isn't just reciting facts but can reason through design trade-offs (e.g., "OIDC vs. SAML", "Private Link vs. Service Endpoints").

```bash
python evaluate.py
```

---

## 📂 Project Structure

```text
Cloud-Sec-Architect-AI/
├── config/                     # Configuration files
│   └── training_config.yaml
├── data/                       # Generated datasets
│   └── architect_training_data.jsonl
├── cloud-architect-v1/         # Saved LoRA Adapters
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── results/                    # Training checkpoints
├── tests/                      # Unit tests
│   └── test_harvester.py
├── harvest_data.py             # Data Engine (Scraping & Cleaning)
├── train.py                    # Training Script (Llama 3.1 + QLoRA)
├── evaluate.py                 # Evaluation Script (Golden Set)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 🧠 Model Architecture Details

| Component | Specification |
| --- | --- |
| **Base Model** | Llama 3.1 8B Instruct |
| **Prompt Format** | Llama 3 (`< |
| **Quantization** | 4-bit NF4 (Normal Float 4) |
| **LoRA Rank (r)** | 32 |
| **Target Modules** | `all-linear` (Self-Attention & MLP layers) |
| **Context Window** | 2048 tokens (Effective training length) |

---

## ⚠️ Disclaimer

This tool is for educational and research purposes. While trained on authoritative sources, AI models can hallucinate. Always verify architectural decisions against official vendor documentation and your organization's compliance requirements before implementation.

---

## 🤝 Contributing

Contributions to expand the `START_URLS` list in the Data Engine or add new "Golden Questions" to the evaluator are highly encouraged! Please open a PR.