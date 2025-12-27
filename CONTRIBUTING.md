# Contributing to cloud-sec-architect-8b

Thank you for your interest in contributing to cloud-sec-architect-8b! This document provides guidelines and best practices for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU with 12GB+ VRAM (for inference) or 24GB+ VRAM (for training)
- CUDA 11.8 or later
- Git

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ftrout/cloud-sec-architect-8b.git
   cd cloud-sec-architect-8b
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Environment details (Python version, GPU, OS)
- Any relevant error messages or logs

### Suggesting Enhancements

We welcome suggestions! Please open an issue with:

- A clear description of the enhancement
- The motivation and use case
- Any implementation ideas you have

### Expanding Training Data Sources

We especially welcome contributions that expand the `START_URLS` list in `harvest_data.py` with:

- Official cloud provider security documentation
- Compliance framework references (NIST, CIS, ISO)
- Kubernetes security guides
- Identity provider documentation (OIDC, SAML)

**Requirements for new data sources:**
- Must be publicly accessible
- Should be authoritative/official documentation
- Must be technical (not marketing content)
- Should have Flesch-Kincaid Grade Level > 8

### Adding Evaluation Questions

Contributions to the "Golden Set" of evaluation questions in `evaluate.py` are encouraged:

- Questions should test complex architectural reasoning
- Include multi-cloud or compliance-focused scenarios
- Cover identity, networking, or IaC security topics

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes following our coding standards**

3. **Run the test suite:**
   ```bash
   pytest
   ```

4. **Run linting and formatting:**
   ```bash
   ruff check .
   black .
   isort .
   mypy .
   ```

5. **Commit with a descriptive message:**
   ```bash
   git commit -m "feat: add new security documentation source"
   ```

6. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Fill out the PR template with:**
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

### Commit Message Format

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Coding Standards

### Python Style Guide

- Follow PEP 8 guidelines
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Use type hints where possible
- Maximum function length: ~50 lines

### Documentation

- All public functions should have docstrings
- Use Google-style docstrings
- Update README.md for user-facing changes
- Update MODEL_CARD.md for model-related changes

### Security

- Never commit API keys, tokens, or credentials
- Use `.env` files for local secrets (excluded from git)
- Review OWASP guidelines for any web-facing code
- Report security vulnerabilities privately

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run only unit tests
pytest -m unit

# Skip slow integration tests
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Mock external APIs and network calls
- Aim for >80% code coverage

### Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_quick_validation():
    pass

@pytest.mark.integration
def test_with_external_service():
    pass

@pytest.mark.slow
def test_full_training_run():
    pass
```

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Join our community discussions

Thank you for helping improve cloud-sec-architect-8b!
