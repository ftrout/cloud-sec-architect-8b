# Technical Requirements: Cloud Security Architect AI
**Project Name:** `cloud-sec-arch-llm`

**Version:** 1.0

**Owner:** Cloud Security Architecture Team

## 1. Executive Summary
The objective is to fine-tune a Large Language Model (LLM) to function as a **Senior Cloud Security Architect**. Unlike generic models (ChatGPT/Claude), this model must specialize in:
1.  **High-Fidelity Technical Guidance:** Specific configurations for AWS, Azure, GCP, Databricks, and Kubernetes.
2.  **Compliance Alignment:** Mapping architectures to frameworks (NIST 800-53, ISO 27001, CIS Benchmarks).
3.  **Context-Awareness:** Distinguishing between "Best Practice" (Vendor Agnostic) and "Implementation Details" (Vendor Specific).

---

## 2. Functional Requirements

### 2.1 Core Capabilities
* **Architectural Design:** The model must generate secure reference architectures (SRAs) based on user inputs (e.g., "Design a HIPAA-compliant data lake on Databricks").
* **Gap Analysis:** The model must identify security gaps in provided snippets of infrastructure-as-code (Terraform/CloudFormation) or description.
* **Compliance Mapping:** Responses must cite relevant controls (e.g., "This requires enabling AWS CloudTrail to satisfy ISO 27001 A.12.4").

### 2.2 Persona & Tone
* **Role:** Senior Security Architect.
* **Tone:** Professional, authoritative, concise, and prescriptive.
* **Safety:** The model must refuse to generate exploit code or "black hat" instructions.

---

## 3. Data Pipeline Requirements ("The Data Engine")

### 3.1 Data Acquisition
* **Sources:** Official documentation roots only.
    * *Allowed:* `docs.aws.amazon.com`, `learn.microsoft.com`, `cloud.google.com`, `kubernetes.io`, `cisecurity.org`.
    * *Excluded:* Blogs, forums (StackOverflow), Reddit, marketing landing pages.
* **Tooling:** `trafilatura` for main-content extraction (ignoring navbars/footers).
* **Volume:** Minimum 2,000 high-quality architectural documentation pages.

### 3.2 Data Engineering & Quality
* **Deduplication:** Implementation of **MinHash LSH** (Locality Sensitive Hashing) to remove semantic duplicates across different URLs.
* **Complexity Filter:** Content must meet a **Flesch-Kincaid Grade Level > 8** to ensure technical depth.
* **Formatting:** Data must be stored in `JSONL` format with `instruction`, `input` (context/source), and `output` fields.

---

## 4. Model Specifications ("The Training Lab")

### 4.1 Base Model Architecture
* **Model ID:** `mistralai/Mistral-7B-Instruct-v0.3` (selected for sliding window attention and high reasoning benchmarks).
* **Context Window:** 4096 tokens (training effective length: 2048 w/ packing).

### 4.2 Fine-Tuning Strategy
* **Method:** **QLoRA** (Quantized Low-Rank Adaptation).
* **Precision:** 4-bit Normal Float (NF4) quantization with `bfloat16` compute.
* **Adapter Config:**
    * Rank (`r`): 64
    * Alpha (`lora_alpha`): 128
    * Target Modules: `all-linear` (Projections: q, k, v, o, gate, up, down).

### 4.3 Training Hyperparameters
* **Optimizer:** `paged_adamw_32bit`.
* **Learning Rate:** `2e-4` with Cosine Scheduler.
* **Batch Strategy:** `packing=True` enabled to maximize token throughput.
* **Gradient Accumulation:** Effective batch size must be $\ge 16$ to ensure stability.

---

## 5. Infrastructure & Environment

### 5.1 Hardware
* **Minimum Training:** 1x NVIDIA GPU with $\ge$ 24GB VRAM (RTX 3090/4090 or A10G).
* **Preferred Training:** 1x NVIDIA A100 (40GB/80GB).
* **Inference:** Can run on CPU (slow) or GPU ($\ge$ 12GB VRAM for 4-bit quantized inference).

### 5.2 Software Stack
* **OS:** Linux (Ubuntu 22.04 LTS).
* **Container:** NVIDIA PyTorch Container (`nvcr.io/nvidia/pytorch:xx.xx-py3`).
* **Tracking:** Weights & Biases (W&B) for loss visualization and artifact versioning.

---

## 6. Evaluation & Acceptance Criteria ("The Judge")

### 6.1 The "Golden Set"
A curated dataset of **50 distinct architectural challenges** with known "perfect" answers (Ground Truth), covering:
* Multi-Cloud Networking (AWS TGW vs. Azure vWAN).
* Identity Management (OIDC, SAML federation).
* Encryption patterns (KMS, CMK vs. Managed Keys).

### 6.2 Acceptance Thresholds
* **Training Loss:** Must converge below `0.8` without overfitting (Validation Loss diverges < 5%).
* **Hallucination Rate:** < 5% on "Golden Set" questions (measured by human review or LLM-Judge).
* **Format Compliance:** 100% of outputs must adhere to the specified Markdown structure.

---

## 7. Roadmap
* **Phase 1 (MVP):** Single-vendor focus (AWS), scraping 500 pages. Train & Validate.
* **Phase 2 (Multi-Cloud):** Integrate Azure/GCP data. Implement `Vendor` tag in input prompt.
* **Phase 3 (RAG Integration):** Connect the model to a Vector Database (Pinecone/Milvus) for real-time documentation lookup (reducing the need for frequent fine-tuning).

### Would you like me to create the "Golden Set" of 10 initial questions for you to test the model against?