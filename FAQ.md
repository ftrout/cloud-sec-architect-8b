# Frequently Asked Questions (FAQ)

## General Questions

### What is cloud-sec-architect-8b?

cloud-sec-architect-8b is a specialized fine-tuned large language model designed to function as a Senior Cloud Security Architect. It provides expert-level guidance on multi-cloud security architecture, compliance frameworks, and infrastructure-as-code security.

### What base model is it built on?

The model is built on Meta's **Llama 3.1 8B Instruct**, fine-tuned using QLoRA (Quantized Low-Rank Adaptation) for efficient training and inference.

### Is the model free to use?

Yes, the model is released under the MIT License. However, you must comply with Meta's Llama 3.1 license agreement for the base model.

### What languages does it support?

Currently, the model only supports **English**. It was trained exclusively on English-language security documentation.

---

## Technical Requirements

### What hardware do I need to run the model?

| Use Case | Minimum GPU VRAM | Recommended GPUs |
|----------|------------------|------------------|
| Inference | 12GB | RTX 3060, RTX 4070, T4 |
| Training | 24GB | RTX 3090, RTX 4090, A10G, A100 |

### What software dependencies are required?

- Python 3.10 or 3.11
- PyTorch 2.2.0+
- CUDA 11.8 or later
- Transformers, PEFT, TRL, bitsandbytes

Install all dependencies with:
```bash
pip install -r requirements.txt
```

### Can I run it on CPU only?

While technically possible, CPU inference is extremely slow and not recommended. The model requires GPU acceleration for practical use.

### Does it work on macOS with Apple Silicon?

The model uses bitsandbytes for 4-bit quantization, which has limited support on Apple Silicon. You may need to use alternative quantization methods or run in full precision (requiring more memory).

---

## Model Usage

### How do I load the model from Hugging Face?

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

### What is the recommended temperature setting?

For security architecture guidance, we recommend:
- **Temperature**: 0.7 (balanced creativity and consistency)
- **Top-p**: 0.9 (nucleus sampling)
- **Max tokens**: 512-1024 (depending on question complexity)

### How do I use the Gradio demo?

```bash
# Run locally
python demo.py

# With custom port
python demo.py --port 8080

# Using HuggingFace Hub model
python demo.py --hf-model fmt0816/cloud-sec-architect-8b
```

The demo will be available at `http://127.0.0.1:7860` by default.

### Can I use this model for commercial purposes?

Yes, under the MIT License. However, ensure you also comply with Meta's Llama 3.1 Community License Agreement.

---

## Training & Fine-tuning

### What data was the model trained on?

The model was trained on curated documentation from:
- AWS Prescriptive Guidance and Well-Architected Framework
- Azure Well-Architected Framework and Security Documentation
- GCP Cloud Architecture Framework
- Kubernetes Official Security Documentation
- MITRE ATT&CK for Cloud
- CIS Controls and Benchmarks

### How do I create my own training data?

Run the data harvester:
```bash
python harvest_data.py
```

This will crawl authorized domains and create `./data/architect_training_data.jsonl`.

### How do I fine-tune the model?

1. Prepare your training data in JSONL format
2. Configure `config/training_config.yaml`
3. Run training:
```bash
python train.py
```

### How long does training take?

Approximately 2-4 hours on a single NVIDIA RTX 3090 (24GB VRAM) for 3 epochs.

### Can I add my own documentation sources?

Yes! Modify the `START_URLS` and `ALLOWED_DOMAINS` lists in `harvest_data.py` to include your own sources. Ensure they are:
- Publicly accessible
- Authoritative/official documentation
- Technical in nature (Flesch-Kincaid Grade Level > 8)

---

## Security & Compliance

### Is the model safe to use in production?

The model is designed for advisory purposes. Always:
1. Verify recommendations against official vendor documentation
2. Have qualified security professionals review AI-generated guidance
3. Test recommendations in non-production environments first

### Can the model generate exploit code?

The model is not designed or intended to generate exploit code. It focuses on defensive security architecture and best practices.

### Does the model store or learn from my queries?

No. The model runs entirely locally or on your own infrastructure. It does not send data externally or learn from your queries.

### Is the training data sanitized?

Yes. The data harvesting pipeline:
- Only collects from authorized public documentation
- Filters for technical depth (Grade Level > 8)
- Removes semantic duplicates
- Does not include proprietary or confidential information

### How do I report a security vulnerability?

Please review our [SECURITY.md](SECURITY.md) file. Report vulnerabilities via:
1. GitHub Security Advisories (preferred)
2. Private issue to the repository maintainer

---

## Troubleshooting

### "CUDA out of memory" error

Try these solutions:
1. Use 4-bit quantization (enabled by default)
2. Reduce batch size in training config
3. Enable gradient checkpointing
4. Use a GPU with more VRAM

### Model produces repetitive or low-quality output

Adjust generation parameters:
- Increase temperature (try 0.8-0.9)
- Lower repetition_penalty (try 1.1)
- Ensure proper chat template is applied

### Adapter not found error

Ensure you've either:
1. Trained the model locally (`python train.py`)
2. Specified the correct HuggingFace Hub model (`--hf-model fmt0816/cloud-sec-architect-8b`)

### Import errors with bitsandbytes

Ensure you have:
1. CUDA toolkit installed
2. Compatible GPU drivers
3. Run: `pip install bitsandbytes --upgrade`

### Slow inference speed

- Ensure you're using GPU (check `device_map="auto"`)
- Use 4-bit quantization
- Consider using Flash Attention 2 (requires compatible GPU)

---

## Model Limitations

### What are the known limitations?

1. **Hallucination Risk**: May generate plausible but incorrect information
2. **Knowledge Cutoff**: Training data has a cutoff date
3. **Context Length**: Optimized for 2048 tokens during training
4. **Vendor Bias**: Some bias toward major cloud providers
5. **English Only**: No multi-language support

### What should I NOT use this model for?

- Making final security decisions without human review
- Replacing certified security professionals
- Real-time threat detection or incident response
- Generating exploit code or malicious content
- Compliance certification without proper auditing

### How often is the model updated?

The model is periodically updated as new security frameworks and cloud services emerge. Check the repository for the latest version.

---

## Integration & Deployment

### Can I deploy this as an API?

Yes! Options include:
1. Gradio (built-in, `demo.py`)
2. FastAPI with custom endpoints
3. vLLM for high-throughput serving
4. Hugging Face Inference Endpoints

### How do I integrate with my CI/CD pipeline?

Use the inference script:
```bash
python scripts/inference.py "Review this Terraform configuration for security issues: ..."
```

Or integrate the `CloudSecurityArchitect` class from `scripts/inference.py` into your Python code.

### Can I use this with LangChain?

Yes, the model is compatible with LangChain's Hugging Face integration:
```python
from langchain_huggingface import HuggingFacePipeline
```

---

## Getting Help

### Where can I get support?

- Open an issue on [GitHub](https://github.com/ftrout/cloud-sec-architect-8b)
- Check existing issues for similar problems
- Review the documentation in this repository

### How can I report bugs?

Open a GitHub issue with:
- Clear description of the problem
- Steps to reproduce
- Environment details (Python version, GPU, OS)
- Relevant error messages or logs
