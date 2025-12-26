---
language:
  - en
license: mit
library_name: transformers
tags:
  - llama
  - llama-3.1
  - cloud-security
  - security-architecture
  - fine-tuned
  - qlora
  - 8b
  - text-generation
  - conversational
pipeline_tag: text-generation
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
model-index:
  - name: cloud-sec-architect-8b
    results: []
datasets:
  - custom
---

# cloud-sec-architect-8b

A specialized fine-tuned LLM designed to function as a **Senior Cloud Security Architect**. Built on Meta's Llama 3.1 8B Instruct, this model provides expert-level guidance on multi-cloud security architecture, compliance frameworks, and infrastructure-as-code security.

## Model Details

### Model Description

**cloud-sec-architect-8b** is a domain-specific fine-tuned model trained on high-fidelity, vendor-agnostic documentation covering cloud security architecture, threat frameworks, and infrastructure-as-code standards. Unlike generic LLMs, this model excels at:

- Designing secure multi-cloud environments (AWS, Azure, GCP)
- Performing security gap analysis
- Mapping technical implementations to compliance frameworks
- Providing actionable security architecture guidance

- **Developed by:** Frank Trout
- **Model type:** Causal Language Model (Fine-tuned)
- **Language(s):** English
- **License:** MIT
- **Finetuned from model:** [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

### Model Sources

- **Repository:** [GitHub - cloud-sec-architect-8b](https://github.com/ftrout/cloud-sec-architect-8b)
- **Base Model:** [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## Uses

### Direct Use

This model is designed for interactive consultation on cloud security architecture topics:

- Cloud security architecture design and review
- Compliance mapping (NIST 800-53, CIS Benchmarks, ISO 27001)
- Identity architecture guidance (OIDC, SAML, OAuth 2.0)
- Infrastructure-as-Code security reviews (Terraform, CloudFormation)
- Kubernetes security best practices
- Multi-cloud security strategy development

### Downstream Use

The model can be integrated into:

- Security architecture review pipelines
- Compliance documentation generators
- Interactive security chatbots
- DevSecOps toolchains

### Out-of-Scope Use

This model should NOT be used for:

- Making final security decisions without human review
- Replacing certified security professionals
- Real-time threat detection or incident response
- Generating exploit code or malicious content

## Bias, Risks, and Limitations

### Known Limitations

- **Hallucination Risk:** Like all LLMs, may generate plausible but incorrect information
- **Knowledge Cutoff:** Training data has a cutoff date; may not reflect latest CVEs or vendor updates
- **Vendor Bias:** While trained on vendor-agnostic sources, some bias toward major cloud providers may exist
- **Context Length:** Optimized for 2048 token context; longer conversations may lose context

### Recommendations

- Always verify architectural recommendations against official vendor documentation
- Use as a starting point for security designs, not as the final authority
- Review generated IaC code before deployment
- Validate compliance mappings with your organization's legal/compliance team

## How to Get Started with the Model

### Installation

```bash
pip install transformers peft bitsandbytes accelerate torch
```

### Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "ftrout/cloud-sec-architect-8b"  # or local path

# Load with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Generate response
def generate_response(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a Senior Cloud Security Architect. Provide detailed, secure, and compliant technical guidance."},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

# Example usage
response = generate_response(
    "What are the key security considerations when designing a multi-region AWS architecture?"
)
print(response)
```

### Gradio Demo

```bash
python demo.py
```

## Training Details

### Training Data

The model was trained on a curated corpus of high-quality technical documentation:

- **AWS:** Prescriptive Guidance, Well-Architected Framework, Security Best Practices
- **Azure:** Well-Architected Framework, Security Documentation
- **GCP:** Cloud Architecture Framework, Security Best Practices
- **Kubernetes:** Official Documentation, Security Guides
- **Compliance:** NIST 800-53, CIS Benchmarks, MITRE ATT&CK for Cloud

Data was processed through a custom pipeline featuring:
- Content extraction using `trafilatura`
- Quality filtering (Flesch-Kincaid Grade Level > 8)
- Semantic deduplication using MinHash LSH (threshold: 0.85)
- Instruction-response pair formatting

### Training Procedure

#### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| Fine-tuning Method | QLoRA (4-bit NF4) |
| LoRA Rank (r) | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| Target Modules | all-linear |
| Epochs | 3 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| Warmup Ratio | 0.03 |
| Max Sequence Length | 2048 |
| Precision | bfloat16 |

#### Hardware

- **Training:** NVIDIA RTX 3090/4090 or A10G (24GB+ VRAM)
- **Inference:** NVIDIA GPU with 12GB+ VRAM

## Evaluation

### Testing Data & Metrics

The model was evaluated against a "Golden Set" of complex architectural questions testing:

- Multi-cloud architecture design
- Compliance framework mapping
- Identity architecture (OIDC vs SAML trade-offs)
- Kubernetes security configurations
- Disaster recovery strategies

### Sample Evaluation Questions

1. "Design a disaster recovery strategy for a multi-region financial application on AWS"
2. "Compare Azure Private Endpoints vs Service Endpoints for a healthcare workload"
3. "What Kubernetes security controls are needed for PCI-DSS compliance?"
4. "When should I use OIDC vs SAML for enterprise identity federation?"

## Technical Specifications

### Model Architecture

- **Architecture:** LlamaForCausalLM with LoRA adapters
- **Parameters:** ~8B (base) + ~50M (LoRA adapters)
- **Quantization:** 4-bit NF4 (Normal Float 4)
- **Context Window:** 2048 tokens (training), 128K tokens (inference capability)
- **Attention:** Flash Attention 2

### Compute Infrastructure

- **Training Framework:** PyTorch + Transformers + PEFT + TRL
- **Experiment Tracking:** Weights & Biases
- **Training Time:** ~2-4 hours on single RTX 3090

## Citation

```bibtex
@misc{cloud-sec-architect-8b,
  author = {Trout, Frank},
  title = {cloud-sec-architect-8b: A Fine-tuned LLM for Cloud Security Architecture},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ftrout/cloud-sec-architect-8b}}
}
```

## Model Card Authors

Frank Trout

## Model Card Contact

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/ftrout/cloud-sec-architect-8b).
