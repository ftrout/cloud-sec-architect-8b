"""
Configuration validation using Pydantic for type safety and validation.
This module provides schema validation for training_config.yaml
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model configuration schema"""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(
        ...,
        description="HuggingFace model ID",
        examples=["meta-llama/Meta-Llama-3.1-8B-Instruct"],
    )
    new_model_name: str = Field(..., description="Name for the fine-tuned model", min_length=1)
    quantization: Literal["nf4", "fp4"] = Field(
        default="nf4", description="Quantization type for 4-bit loading"
    )

    @field_validator("base_model")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate HuggingFace model ID format"""
        if "/" not in v:
            raise ValueError("base_model must be in format 'org/model-name'")
        return v


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration schema"""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(..., ge=1, le=256, description="LoRA rank")
    alpha: int = Field(..., ge=1, description="LoRA alpha scaling parameter")
    dropout: float = Field(..., ge=0.0, le=1.0, description="LoRA dropout rate")
    target_modules: str = Field(
        default="all-linear",
        description="Target modules for LoRA adaptation",
    )

    @field_validator("r", "alpha")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure positive values"""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha_ratio(cls, v: int, info) -> int:
        """Validate alpha is reasonable relative to rank"""
        # Note: in pydantic v2, we access other fields via info.data
        r_value = info.data.get("r")
        if r_value and v > r_value * 10:
            raise ValueError(f"alpha ({v}) should typically be <= 10x rank ({r_value})")
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration schema"""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    epochs: int = Field(..., ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(..., ge=1, le=128, description="Per-device batch size")
    grad_accum_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    learning_rate: float = Field(..., gt=0.0, le=1.0, description="Learning rate")
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=0.5, description="Warmup ratio")
    max_seq_length: int = Field(
        default=2048, ge=128, le=8192, description="Maximum sequence length"
    )
    logging_steps: int = Field(default=10, ge=1, description="Logging frequency")
    eval_steps: int = Field(default=100, ge=1, description="Evaluation frequency")
    save_steps: int = Field(default=500, ge=1, description="Checkpoint save frequency")
    output_dir: str = Field(default="./results", description="Output directory path")

    @model_validator(mode="after")
    def validate_effective_batch_size(self):
        """Validate effective batch size isn't too large"""
        effective_batch = self.batch_size * self.grad_accum_steps

        if effective_batch > 128:
            raise ValueError(
                f"Effective batch size ({effective_batch}) = "
                f"batch_size ({self.batch_size}) x grad_accum_steps ({self.grad_accum_steps}) "
                f"should not exceed 128 for stability"
            )
        return self

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Warn if learning rate is unusually high or low"""
        if v > 1e-3:
            raise ValueError(
                f"Learning rate {v} is very high for fine-tuning. " f"Typically use 1e-5 to 5e-4"
            )
        if v < 1e-6:
            raise ValueError(f"Learning rate {v} is very low and may not train effectively")
        return v


class SystemConfig(BaseModel):
    """System and logging configuration schema"""

    model_config = ConfigDict(extra="forbid")

    use_wandb: bool = Field(default=True, description="Enable Weights & Biases logging")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )


class Config(BaseModel):
    """Root configuration schema"""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    system: SystemConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate configuration from YAML file

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return self.model_dump()


def load_config(config_path: str = "config/training_config.yaml") -> Config:
    """Convenience function to load and validate configuration

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Validated Config object

    Example:
        >>> config = load_config()
        >>> print(config.training.epochs)
        3
    """
    return Config.from_yaml(config_path)


# Example usage and validation
if __name__ == "__main__":
    try:
        config = load_config()
        print("✅ Configuration validation successful!")
        print("\nConfiguration summary:")
        print(f"  Model: {config.model.base_model}")
        print(f"  LoRA rank: {config.lora.r}")
        print(f"  Epochs: {config.training.epochs}")
        print(
            f"  Effective batch size: {config.training.batch_size * config.training.grad_accum_steps}"
        )
        print(f"  Learning rate: {config.training.learning_rate}")
    except Exception as e:
        print("❌ Configuration validation failed:")
        print(f"  {e}")
        exit(1)
