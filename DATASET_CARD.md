---
language:
  - en
license: mit
task_categories:
  - text-generation
  - question-answering
tags:
  - cloud-security
  - security-architecture
  - cybersecurity
  - aws
  - azure
  - gcp
  - kubernetes
  - compliance
  - nist
  - cis-benchmarks
  - fine-tuning
  - instruction-tuning
size_categories:
  - 1K<n<10K
---

# Cloud Security Architecture Training Dataset

## Dataset Description

This dataset is specifically curated for fine-tuning large language models to provide expert-level cloud security architecture guidance. It contains instruction-response pairs derived from authoritative cloud security documentation.

### Dataset Summary

The dataset consists of high-quality, technical instruction-response pairs covering:

- Multi-cloud security architecture (AWS, Azure, GCP)
- Kubernetes and container security
- Compliance frameworks (NIST 800-53, CIS Benchmarks)
- Identity and access management (OIDC, SAML, OAuth 2.0)
- Infrastructure-as-Code security (Terraform, CloudFormation)
- Zero Trust architecture principles

### Supported Tasks

- **Text Generation**: Generate detailed cloud security architecture guidance
- **Question Answering**: Answer technical questions about cloud security best practices
- **Instruction Following**: Follow complex security architecture design instructions

### Languages

English (en)

## Dataset Structure

### Data Instances

Each instance contains three fields:

```json
{
  "instruction": "As a Senior Cloud Security Architect, explain the concepts regarding docs.aws.amazon.com.",
  "input": "Source: https://docs.aws.amazon.com/prescriptive-guidance/...",
  "output": "Detailed technical response about cloud security concepts..."
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | The task instruction or question for the model |
| `input` | string | Additional context including the source URL |
| `output` | string | The expected response containing technical guidance |

### Data Splits

The dataset is provided as a single training file. During fine-tuning, it is typically split:

| Split | Percentage | Purpose |
|-------|------------|---------|
| Train | 90% | Model training |
| Validation | 10% | Evaluation during training |

## Dataset Creation

### Curation Rationale

The dataset was created to address the need for specialized LLMs capable of providing expert-level cloud security architecture guidance. Generic LLMs often lack the depth and specificity required for enterprise security architecture decisions.

### Source Data

#### Initial Data Collection

Data was collected from authoritative public documentation sources:

| Source | Domain | Content Type |
|--------|--------|--------------|
| AWS | docs.aws.amazon.com | Prescriptive Guidance, Well-Architected Framework |
| Azure | learn.microsoft.com | Well-Architected Framework, Security Documentation |
| GCP | cloud.google.com | Cloud Architecture Framework, Security Best Practices |
| Kubernetes | kubernetes.io | Official Security Documentation |
| MITRE | attack.mitre.org | ATT&CK for Cloud Matrix |
| CIS | cisecurity.org | CIS Controls and Benchmarks |

#### Data Collection Process

The data was collected using a custom harvesting pipeline (`harvest_data.py`) with the following characteristics:

1. **Web Scraping**: Used `trafilatura` for clean content extraction
2. **Quality Filtering**: Applied Flesch-Kincaid Grade Level filter (> 8) to ensure technical depth
3. **Deduplication**: Used MinHash LSH (threshold: 0.85) to remove semantic duplicates
4. **Chunking**: Split content into ~2000 character chunks for optimal training
5. **Instruction Generation**: Applied diverse instruction templates for varied training

### Annotations

#### Annotation Process

The dataset uses automated instruction generation with six diverse templates:

1. Conceptual explanation requests
2. Best practices queries
3. Architecture design prompts
4. Compliance analysis requests
5. Risk assessment questions
6. Zero Trust implementation guidance

#### Who are the annotators?

The instruction templates were designed by cloud security professionals. The response content is derived directly from authoritative documentation without manual annotation.

## Considerations for Using the Data

### Social Impact

This dataset aims to democratize access to cloud security knowledge, enabling:

- Faster security architecture reviews
- Improved compliance posture for organizations
- Enhanced security awareness among developers and architects

### Discussion of Biases

**Known Biases:**

- **Vendor Focus**: Emphasis on major cloud providers (AWS, Azure, GCP)
- **English Only**: Limited to English-language documentation
- **Enterprise Focus**: Content skewed toward enterprise use cases
- **Temporal Bias**: Reflects documentation available at collection time

### Limitations

- Does not include proprietary or confidential security documentation
- May not reflect the latest CVEs or security advisories
- Content limited to publicly available documentation
- No coverage of emerging cloud-native security tools

### Recommendations

- Use in conjunction with up-to-date vendor documentation
- Verify all recommendations with your security team
- Consider organizational context when applying guidance
- Supplement with domain-specific security requirements

## Additional Information

### Dataset Curators

Frank Trout

### Licensing Information

This dataset is licensed under the MIT License. Note that the source documentation may have its own licensing terms.

### Citation Information

```bibtex
@misc{cloud-sec-architect-dataset,
  author = {Trout, Frank},
  title = {Cloud Security Architecture Training Dataset},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/fmt0816/cloud-sec-architect-data}}
}
```

### Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/ftrout/cloud-sec-architect-8b).
