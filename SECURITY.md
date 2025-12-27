# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories**: Use the [Security tab](https://github.com/ftrout/cloud-sec-architect-8b/security/advisories/new) to privately report a vulnerability.

2. **Email**: Send details to the repository maintainer (see GitHub profile for contact information).

### What to Include

When reporting a vulnerability, please include:

- Type of vulnerability (e.g., prompt injection, data leakage, dependency vulnerability)
- Full path to the affected source file(s)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact assessment and potential attack scenarios

### Response Timeline

- **Initial Response**: Within 48 hours of report submission
- **Status Update**: Within 7 days with assessment of the vulnerability
- **Resolution**: Depending on severity, typically within 30 days

### Scope

This security policy covers:

- The training pipeline (`train.py`, `harvest_data.py`)
- Inference scripts (`demo.py`, `scripts/inference.py`)
- Configuration validation (`config_validation.py`)
- Dependencies specified in `requirements.txt` and `pyproject.toml`

### Out of Scope

The following are generally out of scope:

- Issues in the base Llama 3.1 model (report to Meta)
- Issues in third-party dependencies (report to respective maintainers, but do notify us)
- Model hallucinations or incorrect outputs (this is an inherent LLM limitation)
- Prompt injection attacks that don't lead to system compromise

## Security Best Practices for Users

### API Keys and Credentials

- Never commit `.env` files or API keys to version control
- Use the provided `.env.example` as a template
- Rotate credentials regularly

### Model Deployment

- Run inference in isolated environments
- Implement rate limiting for public-facing deployments
- Monitor for unusual query patterns
- Always validate and sanitize user inputs before passing to the model

### Data Handling

- The data harvester only collects from authorized public documentation
- Review harvested data before training
- Do not train on proprietary or sensitive documentation without authorization

## Dependency Security

We use the following tools to maintain dependency security:

- **pip-audit**: Checks for known vulnerabilities in dependencies
- **Bandit**: Static security analysis for Python code
- **GitHub Dependabot**: Automated dependency updates

Run security checks locally:

```bash
# Install security tools
pip install pip-audit bandit

# Check dependencies for vulnerabilities
pip-audit --requirement requirements.txt

# Run static security analysis
bandit -r . -c pyproject.toml
```

