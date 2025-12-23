# Professional Grade Code Review
## Cloud-Sec-Architect-AI

**Review Date:** 2025-12-23
**Reviewed By:** Claude Code
**Codebase Version:** commit a76d16e

---

## Executive Summary

Your ML pipeline demonstrates solid fundamentals with recent improvements in configuration management. However, to reach professional/production-grade quality, the codebase needs improvements in:

1. **Code Organization & Architecture** (High Priority)
2. **Testing & Quality Assurance** (High Priority)
3. **Error Handling & Resilience** (High Priority)
4. **Documentation & Type Safety** (Medium Priority)
5. **CI/CD & Automation** (Medium Priority)
6. **Security & Secrets Management** (Medium Priority)
7. **Observability & Monitoring** (Low Priority)

**Current Maturity Level:** Research/Prototype → **Target:** Production-Ready

---

## Critical Issues (Fix Immediately)

### 1. Broken Test Imports (1_harvest_data.py:3)
**Location:** `tests/test_harvester.py:3`

```python
# BROKEN - File is named 1_harvest_data.py, not harvester.py
from harvester import DataHarvester
```

**Issue:** Tests will fail to import. The test references `harvester` but the actual file is `1_harvest_data.py`.

**Fix:**
```python
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from harvest_data import DataHarvester  # After renaming files
```

**Impact:** All tests currently fail to run, meaning you have zero test coverage validation.

---

### 2. README Documentation Mismatch
**Location:** `README.md:67-98`

**Issue:** Documentation references non-existent files:
- `1_architect_harvester.py` (actual: `1_harvest_data.py`)
- `2_train_architect.py` (actual: `2_train.py`)
- `3_evaluate_architect.py` (actual: `3_evaluate.py`)
- `./data_architect/` (actual: `./data/`)
- `./results_architect/` (actual: `./results/`)

**Impact:** Users following the README will encounter errors immediately.

---

### 3. Inconsistent Configuration Management
**Location:** `3_evaluate.py:5-6`

```python
# HARDCODED - Should use config/training_config.yaml
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "cloud-architect-v1"
```

**Issue:** After refactoring to YAML config, evaluation script still uses hardcoded values. Creates drift risk.

**Fix:** Load from `config/training_config.yaml` like `2_train.py` does.

---

## High Priority Improvements

### 4. Code Organization & Project Structure

**Current State:**
```
/
├── 1_harvest_data.py      # Pipeline scripts in root
├── 2_train.py
├── 3_evaluate.py
```

**Professional Standard:**
```
/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── harvester.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── config.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── validation.py
├── scripts/
│   ├── run_harvester.py
│   ├── run_training.py
│   └── run_evaluation.py
├── tests/
│   ├── __init__.py
│   ├── test_harvester.py
│   ├── test_trainer.py
│   └── test_evaluator.py
├── config/
│   └── training_config.yaml
├── setup.py or pyproject.toml
└── README.md
```

**Benefits:**
- Installable package (`pip install -e .`)
- Proper import paths
- Cleaner separation of concerns
- Easier testing

---

### 5. Comprehensive Error Handling

**Location:** `1_harvest_data.py:162-163`

```python
# TOO GENERIC
except Exception as e:
    logger.error(f"Error processing {url}: {e}")
```

**Issues:**
- Catches ALL exceptions (including KeyboardInterrupt, SystemExit)
- No retry logic for transient network failures
- No error categorization

**Professional Fix:**
```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

MAX_RETRIES = 3
BACKOFF_FACTOR = 2

for attempt in range(MAX_RETRIES):
    try:
        downloaded = trafilatura.fetch_url(url, config=self.traf_config)
        if not downloaded:
            logger.warning(f"Empty response from {url}")
            continue
        break
    except (RequestException, Timeout, ConnectionError) as e:
        wait_time = BACKOFF_FACTOR ** attempt
        logger.warning(f"Network error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait_time)
        else:
            logger.error(f"Failed after {MAX_RETRIES} attempts: {url}")
            continue
    except ValueError as e:
        logger.error(f"Invalid URL or parsing error: {e}")
        continue
    except Exception as e:
        logger.critical(f"Unexpected error processing {url}: {e}", exc_info=True)
        continue
```

---

### 6. Configuration Validation

**Location:** `2_train.py:18-19`

```python
# NO VALIDATION
with open("config/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)
```

**Issues:**
- No schema validation
- Fails late if required keys missing
- No type checking for values

**Professional Fix (using Pydantic):**

```python
from pydantic import BaseModel, Field, validator
from typing import Literal

class ModelConfig(BaseModel):
    base_model: str = Field(..., description="HuggingFace model ID")
    new_model_name: str
    quantization: Literal["nf4", "fp4"]

class LoRAConfig(BaseModel):
    r: int = Field(gt=0, le=256)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, le=1)
    target_modules: str

class TrainingConfig(BaseModel):
    seed: int = 42
    epochs: int = Field(gt=0, le=100)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    # ... other fields

class Config(BaseModel):
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig

    @validator('training')
    def validate_batch_size(cls, v, values):
        if v.batch_size * v.grad_accum_steps > 128:
            raise ValueError("Effective batch size too large")
        return v

# Usage
def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)  # Raises ValidationError with clear messages
```

**Add to requirements.txt:**
```
pydantic>=2.0.0
```

---

### 7. Comprehensive Testing

**Current Coverage:** ~5% (2 basic tests)

**Professional Test Suite:**

```python
# tests/test_harvester.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.data.harvester import DataHarvester

class TestDataHarvester:
    @pytest.fixture
    def harvester(self, tmp_path):
        """Fixture with temporary directory"""
        with patch('src.data.harvester.OUTPUT_DIR', str(tmp_path)):
            yield DataHarvester()

    def test_minhash_generation(self, harvester):
        """Test MinHash signature generation"""
        text1 = "Sample security documentation"
        text2 = "Sample security documentation"
        text3 = "Different content entirely"

        hash1 = harvester._get_minhash(text1)
        hash2 = harvester._get_minhash(text2)
        hash3 = harvester._get_minhash(text3)

        # Identical text should produce identical hashes
        assert hash1.jaccard(hash2) == 1.0
        # Different text should produce different hashes
        assert hash1.jaccard(hash3) < 0.5

    def test_quality_gates(self, harvester):
        """Test text quality filtering"""
        short_text = "Too short"
        assert len(short_text) < 500  # Should be rejected

        low_grade = "See spot run. Run spot run. " * 100  # Low FK grade
        from textstat import textstat
        assert textstat.flesch_kincaid_grade(low_grade) < 8

    def test_checkpoint_save_load(self, harvester, tmp_path):
        """Test checkpoint persistence"""
        harvester.visited = {"url1", "url2"}
        harvester.collected_count = 42
        harvester._save_checkpoint()

        # Create new harvester and load checkpoint
        new_harvester = DataHarvester()
        assert new_harvester.collected_count == 42
        assert "url1" in new_harvester.visited

    @patch('trafilatura.fetch_url')
    def test_network_error_handling(self, mock_fetch, harvester):
        """Test resilience to network failures"""
        mock_fetch.side_effect = ConnectionError("Network down")

        # Should log error but not crash
        harvester.run()  # Should complete without raising

    def test_url_domain_filtering(self, harvester):
        """Test allowed domains enforcement"""
        from urllib.parse import urlparse

        malicious_url = "https://evil.com/phishing"
        domain = urlparse(malicious_url).netloc
        assert domain not in harvester.ALLOWED_DOMAINS

# tests/test_trainer.py
import pytest
from src.training.trainer import set_seed, formatting_prompts_func

def test_seed_reproducibility():
    """Test that seeding produces deterministic results"""
    import torch
    import random

    set_seed(42)
    r1 = random.random()
    t1 = torch.rand(1).item()

    set_seed(42)
    r2 = random.random()
    t2 = torch.rand(1).item()

    assert r1 == r2
    assert t1 == t2

def test_prompt_formatting():
    """Test Llama 3.1 chat template formatting"""
    batch = {
        'instruction': ["Explain IAM"],
        'input': ["Source: docs.aws.amazon.com"],
        'output': ["IAM is..."]
    }

    formatted = formatting_prompts_func(batch)
    assert len(formatted) == 1
    assert "<|begin_of_text|>" in formatted[0]
    assert "Senior Cloud Security Architect" in formatted[0]
    assert "<|eot_id|>" in formatted[0]

# tests/test_config.py
def test_config_schema_validation():
    """Test config validation catches errors"""
    from src.training.config import Config
    from pydantic import ValidationError

    invalid_config = {
        "model": {"base_model": "test", "quantization": "invalid"},
        "lora": {"r": -1},  # Invalid: negative rank
        "training": {"epochs": 0}  # Invalid: zero epochs
    }

    with pytest.raises(ValidationError) as exc_info:
        Config(**invalid_config)

    assert "r" in str(exc_info.value)
```

**Add pytest configuration:**

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "--verbose",
    "--strict-markers",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing:skip-covered"
]
```

**Add to requirements.txt:**
```
pytest-cov>=4.0.0
pytest-mock>=3.11.0
```

---

### 8. Type Hints & Static Analysis

**Current State:** Partial type hints

**Professional Standard:**

```python
# 1_harvest_data.py with full type hints
from typing import List, Set, Dict, Optional, Tuple
from pathlib import Path

class DataHarvester:
    def __init__(self) -> None:
        self.lsh: MinHashLSH = MinHashLSH(threshold=0.85, num_perm=128)
        self.visited: Set[str] = set()
        self.queue: List[str] = list(START_URLS)
        self.collected_count: int = 0
        self.traf_config: Any = use_config()  # trafilatura doesn't export types
        self._load_checkpoint()

    def _get_minhash(self, text: str) -> MinHash:
        """Generate MinHash signature for deduplication.

        Args:
            text: Input text to hash

        Returns:
            MinHash signature with 128 permutations
        """
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    def _is_duplicate(self, text: str, doc_id: str) -> bool:
        """Check if text is semantically duplicate.

        Args:
            text: Text to check
            doc_id: Unique identifier for this document

        Returns:
            True if duplicate found in LSH index, False otherwise
        """
        m = self._get_minhash(text)
        if len(self.lsh.query(m)) > 0:
            return True
        self.lsh.insert(doc_id, m)
        return False
```

**Add mypy configuration:**

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = [
    "trafilatura.*",
    "datasketch.*",
    "textstat.*",
]
ignore_missing_imports = true
```

---

## Medium Priority Improvements

### 9. Logging Strategy

**Issues:**
- Mixed `print()` and `logger` usage
- No structured logging for metrics
- No log rotation

**Professional Fix:**

```python
# src/utils/logging.py
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class StructuredLogger:
    """JSON-structured logging for ML pipelines"""

    def __init__(self, name: str, log_dir: Path = Path("./logs")):
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON file handler
        json_handler = logging.FileHandler(
            log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )
        json_handler.setFormatter(self.JSONFormatter())

        # Console handler (human-readable)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        )

        self.logger.addHandler(json_handler)
        self.logger.addHandler(console_handler)

    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_obj = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            if record.exc_info:
                log_obj["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_obj)

    def log_metric(self, metric_name: str, value: float, step: int = None, **kwargs):
        """Log a metric with optional metadata"""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "step": step,
            **kwargs
        }
        self.logger.info(f"METRIC: {json.dumps(metric_data)}")

# Usage in harvester
logger = StructuredLogger("harvester")
logger.log_metric("documents_collected", self.collected_count, url=url)
```

**Replace in 2_train.py:**
```python
# BEFORE
print("Starting Professional Training Pipeline...")

# AFTER
from src.utils.logging import StructuredLogger
logger = StructuredLogger("trainer")
logger.logger.info("Starting training pipeline", extra={
    "config": config,
    "gpu_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count()
})
```

---

### 10. CI/CD Pipeline

**Missing:** Automated testing, linting, security scanning

**Professional Standard:**

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, claude/*]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install ruff mypy black isort
          pip install -r requirements.txt

      - name: Run ruff
        run: ruff check .

      - name: Run mypy
        run: mypy src/

      - name: Check formatting (black)
        run: black --check .

      - name: Check import sorting
        run: isort --check-only .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pytest pytest-cov pytest-mock
          pip install -r requirements.txt

      - name: Run tests
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Bandit security scan
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-report.json

      - name: Run pip-audit
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.txt

  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for outdated dependencies
        run: |
          pip install pip-check
          pip-check
```

**Pre-commit hooks:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
      - id: check-json
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
```

**Setup:**
```bash
pip install pre-commit
pre-commit install
```

---

### 11. Secrets Management

**Current Issues:**
- No `.env` file support shown
- HuggingFace token management unclear
- No secrets validation

**Professional Fix:**

```python
# src/utils/secrets.py
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

class SecretsManager:
    """Centralized secrets management"""

    def __init__(self, env_file: Optional[Path] = None):
        if env_file is None:
            env_file = Path(".env")

        if env_file.exists():
            load_dotenv(env_file)

    def get_hf_token(self) -> str:
        """Get HuggingFace token with validation"""
        token = os.getenv("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError(
                "HUGGING_FACE_TOKEN not found. "
                "Set it via: export HUGGING_FACE_TOKEN=<your-token> "
                "or add to .env file"
            )
        if not token.startswith("hf_"):
            raise ValueError("Invalid HuggingFace token format")
        return token

    def get_wandb_key(self) -> Optional[str]:
        """Get W&B API key (optional)"""
        return os.getenv("WANDB_API_KEY")

# Usage in 2_train.py
from src.utils.secrets import SecretsManager

secrets = SecretsManager()
hf_token = secrets.get_hf_token()

# Login programmatically
from huggingface_hub import login
login(token=hf_token)
```

**Create `.env.example`:**
```bash
# .env.example - Copy to .env and fill in values
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx
LOG_LEVEL=INFO
```

**Update `.gitignore`:**
```
# Secrets
.env
.env.local
*.key
*.pem
```

---

### 12. Dependency Management

**Issues:**
- No `setup.py` or `pyproject.toml`
- Can't install as package
- No version constraints justification

**Professional Fix:**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cloud-sec-architect-ai"
version = "0.1.0"
description = "Fine-tuned LLM for Cloud Security Architecture"
authors = [{name = "Frank Trout", email = "your-email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.10,<3.12"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "torch==2.2.0",
    "transformers==4.40.0",
    "peft==0.10.0",
    "trl==0.8.0",
    "bitsandbytes==0.43.0",
    "accelerate==0.28.0",
    "datasets>=2.17.0",
    "trafilatura==1.8.0",
    "datasketch==1.6.4",
    "textstat==0.7.3",
    "scipy==1.12.0",
    "wandb==0.16.4",
    "mlflow==2.11.1",
    "pyyaml==6.0.1",
    "python-dotenv==1.0.1",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.11.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "black>=23.0.0",
    "isort>=5.13.0",
    "pre-commit>=3.5.0",
    "ipython>=8.20.0",
]
eval = [
    "rouge-score==0.1.2",
    "bert-score==0.3.13",
]

[project.scripts]
harvest-data = "src.data.harvester:main"
train-model = "src.training.trainer:main"
evaluate-model = "src.evaluation.evaluator:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.black]
line-length = 100
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
target-version = "py310"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
```

**Install as editable package:**
```bash
pip install -e .  # Development mode
pip install -e ".[dev]"  # With dev dependencies
```

---

## Low Priority (Nice-to-Have)

### 13. Performance Monitoring

```python
# src/utils/profiling.py
import time
import functools
from typing import Callable
from src.utils.logging import StructuredLogger

logger = StructuredLogger("profiling")

def profile(func: Callable):
    """Decorator to profile function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        logger.log_metric(
            metric_name=f"{func.__module__}.{func.__name__}_duration",
            value=elapsed,
            unit="seconds"
        )
        return result
    return wrapper

# Usage
@profile
def run(self):
    # ... harvesting logic
```

---

### 14. Data Validation & Quality Metrics

```python
# src/data/quality.py
from typing import Dict, List
from collections import Counter
import json

class DataQualityReport:
    """Generate quality report for training data"""

    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]

    def generate_report(self) -> Dict:
        """Comprehensive data quality analysis"""
        report = {
            "total_samples": len(self.data),
            "avg_output_length": self._avg_length("output"),
            "avg_instruction_length": self._avg_length("instruction"),
            "unique_sources": len(set(d["input"] for d in self.data)),
            "instruction_diversity": self._diversity("instruction"),
            "domain_distribution": self._domain_distribution(),
        }
        return report

    def _avg_length(self, field: str) -> float:
        lengths = [len(d[field].split()) for d in self.data]
        return sum(lengths) / len(lengths)

    def _diversity(self, field: str) -> float:
        """Shannon entropy for diversity"""
        import math
        counter = Counter(d[field] for d in self.data)
        total = sum(counter.values())
        entropy = -sum((count/total) * math.log2(count/total)
                      for count in counter.values())
        return entropy

    def _domain_distribution(self) -> Dict[str, int]:
        """Count samples per domain"""
        from urllib.parse import urlparse
        domains = []
        for d in self.data:
            url = d["input"].replace("Source: ", "")
            domain = urlparse(url).netloc
            domains.append(domain)
        return dict(Counter(domains))

# Run after harvesting
from src.data.quality import DataQualityReport
report = DataQualityReport("./data/architect_training_data.jsonl")
print(json.dumps(report.generate_report(), indent=2))
```

---

### 15. Model Versioning & Registry

```python
# src/training/model_registry.py
import mlflow
from pathlib import Path
from typing import Dict, Any

class ModelRegistry:
    """MLflow-based model versioning"""

    def __init__(self, experiment_name: str = "cloud-sec-architect"):
        mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        model_path: Path
    ):
        """Log complete training run to MLflow"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self._flatten_dict(config))

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifacts
            mlflow.log_artifacts(str(model_path), artifact_path="model")

            # Tag with metadata
            mlflow.set_tags({
                "framework": "transformers + peft",
                "base_model": config["model"]["base_model"],
                "model_type": "qlora",
            })

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dict for MLflow params"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

# Usage in 2_train.py
registry = ModelRegistry()
registry.log_training_run(
    config=config,
    metrics={
        "final_train_loss": trainer.state.log_history[-1]["loss"],
        "final_eval_loss": trainer.state.log_history[-1]["eval_loss"],
    },
    model_path=Path(config["model"]["new_model_name"])
)
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix test imports and rename files consistently
- [ ] Update README.md with correct file paths
- [ ] Migrate `3_evaluate.py` to use YAML config
- [ ] Add basic error handling with retries

### Phase 2: Code Quality (Week 2)
- [ ] Restructure into `src/` package layout
- [ ] Add comprehensive type hints
- [ ] Implement configuration validation (Pydantic)
- [ ] Add pre-commit hooks

### Phase 3: Testing & CI (Week 3)
- [ ] Expand test suite to 80%+ coverage
- [ ] Set up GitHub Actions CI pipeline
- [ ] Add security scanning (Bandit, pip-audit)
- [ ] Configure mypy strict mode

### Phase 4: Production Readiness (Week 4)
- [ ] Implement structured logging
- [ ] Add secrets management
- [ ] Create `pyproject.toml` for package distribution
- [ ] Add data quality validation
- [ ] Set up MLflow model registry

---

## Code Quality Metrics

### Before Improvements
| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | ~5% | 80%+ |
| Type Hint Coverage | ~40% | 95%+ |
| Linting Compliance | Unknown | 100% |
| Documentation | Moderate | Comprehensive |
| Modularity Score | Low | High |
| Error Handling | Basic | Robust |

### After Improvements
- **Installable Package**: `pip install -e .`
- **Automated Testing**: CI runs on every commit
- **Type Safety**: mypy strict mode passes
- **Security**: No critical vulnerabilities
- **Maintainability**: Clear separation of concerns

---

## Quick Wins (Do These First)

1. **Fix broken imports** (5 minutes)
2. **Update README paths** (10 minutes)
3. **Add `.env` support** (15 minutes)
4. **Install pre-commit hooks** (10 minutes)
5. **Add retry logic to harvester** (20 minutes)
6. **Create GitHub Actions CI** (30 minutes)

---

## References & Best Practices

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [ML Engineering Best Practices](https://ml-ops.org/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Best Practices](https://huggingface.co/docs/peft/index)
- [Python Packaging Guide](https://packaging.python.org/)

---

## Summary

Your codebase shows strong ML engineering fundamentals, but needs professional software engineering practices to be production-ready. Focus on:

1. **Immediate**: Fix broken tests and documentation
2. **Short-term**: Add error handling, validation, and testing
3. **Long-term**: Restructure for maintainability and scalability

The good news: Most improvements are incremental and won't require rewriting core logic. Start with quick wins and build momentum.

**Estimated effort to reach production-grade: 3-4 weeks part-time**

Let me know which areas you'd like to tackle first, and I can provide detailed implementation guidance!
