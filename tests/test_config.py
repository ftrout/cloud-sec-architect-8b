"""
Tests for configuration validation
"""

import pytest
import sys
import os
from pathlib import Path
from pydantic import ValidationError

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_validation import (
    Config,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    SystemConfig,
    load_config
)


class TestModelConfig:
    """Tests for ModelConfig validation"""

    def test_valid_model_config(self):
        """Test valid model configuration"""
        config = ModelConfig(
            base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            new_model_name="test-model",
            quantization="nf4"
        )
        assert config.base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert config.quantization == "nf4"

    def test_invalid_model_id_format(self):
        """Test that model ID without '/' is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                base_model="invalid-model-id",
                new_model_name="test",
                quantization="nf4"
            )
        assert "base_model must be in format" in str(exc_info.value)

    def test_invalid_quantization_type(self):
        """Test that invalid quantization type is rejected"""
        with pytest.raises(ValidationError):
            ModelConfig(
                base_model="org/model",
                new_model_name="test",
                quantization="invalid"  # Only nf4 or fp4 allowed
            )


class TestLoRAConfig:
    """Tests for LoRAConfig validation"""

    def test_valid_lora_config(self):
        """Test valid LoRA configuration"""
        config = LoRAConfig(r=32, alpha=64, dropout=0.05, target_modules="all-linear")
        assert config.r == 32
        assert config.alpha == 64
        assert config.dropout == 0.05

    def test_negative_rank_rejected(self):
        """Test that negative rank is rejected"""
        with pytest.raises(ValidationError):
            LoRAConfig(r=-1, alpha=64, dropout=0.05)

    def test_rank_too_large(self):
        """Test that rank > 256 is rejected"""
        with pytest.raises(ValidationError):
            LoRAConfig(r=512, alpha=64, dropout=0.05)

    def test_dropout_out_of_range(self):
        """Test that dropout outside [0, 1] is rejected"""
        with pytest.raises(ValidationError):
            LoRAConfig(r=32, alpha=64, dropout=1.5)

    def test_alpha_too_large_warning(self):
        """Test that alpha > 10x rank triggers warning"""
        with pytest.raises(ValidationError) as exc_info:
            LoRAConfig(r=32, alpha=1000, dropout=0.05)
        assert "10x rank" in str(exc_info.value)


class TestTrainingConfig:
    """Tests for TrainingConfig validation"""

    def test_valid_training_config(self):
        """Test valid training configuration"""
        config = TrainingConfig(
            seed=42,
            epochs=3,
            batch_size=4,
            grad_accum_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            max_seq_length=2048,
            logging_steps=10,
            eval_steps=100,
            save_steps=500,
            output_dir="./results"
        )
        assert config.epochs == 3
        assert config.learning_rate == 2e-4

    def test_effective_batch_size_too_large(self):
        """Test that effective batch size > 128 is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                epochs=3,
                batch_size=64,
                grad_accum_steps=4,  # 64 * 4 = 256 > 128
                learning_rate=2e-4
            )
        assert "should not exceed 128" in str(exc_info.value)

    def test_learning_rate_too_high(self):
        """Test that very high learning rate is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                epochs=3,
                batch_size=4,
                grad_accum_steps=4,
                learning_rate=1e-2  # Too high for fine-tuning
            )
        assert "very high" in str(exc_info.value).lower()

    def test_learning_rate_too_low(self):
        """Test that very low learning rate is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                epochs=3,
                batch_size=4,
                grad_accum_steps=4,
                learning_rate=1e-7  # Too low
            )
        assert "very low" in str(exc_info.value).lower()

    def test_epochs_out_of_range(self):
        """Test that epochs outside valid range is rejected"""
        with pytest.raises(ValidationError):
            TrainingConfig(
                epochs=0,  # Must be >= 1
                batch_size=4,
                learning_rate=2e-4
            )


class TestSystemConfig:
    """Tests for SystemConfig validation"""

    def test_valid_system_config(self):
        """Test valid system configuration"""
        config = SystemConfig(use_wandb=True, log_level="INFO")
        assert config.use_wandb is True
        assert config.log_level == "INFO"

    def test_invalid_log_level(self):
        """Test that invalid log level is rejected"""
        with pytest.raises(ValidationError):
            SystemConfig(log_level="INVALID")


class TestFullConfig:
    """Tests for complete configuration"""

    def test_load_valid_config_file(self):
        """Test loading valid configuration file"""
        # This assumes config/training_config.yaml exists and is valid
        try:
            config = load_config("config/training_config.yaml")
            assert config.model.base_model is not None
            assert config.training.epochs > 0
            assert config.lora.r > 0
        except FileNotFoundError:
            pytest.skip("Configuration file not found")

    def test_config_to_dict(self):
        """Test exporting configuration to dictionary"""
        config = Config(
            model=ModelConfig(
                base_model="org/model",
                new_model_name="test",
                quantization="nf4"
            ),
            lora=LoRAConfig(r=32, alpha=64, dropout=0.05),
            training=TrainingConfig(
                epochs=3,
                batch_size=4,
                grad_accum_steps=4,
                learning_rate=2e-4
            ),
            system=SystemConfig(use_wandb=True, log_level="INFO")
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "lora" in config_dict

    def test_extra_fields_rejected(self):
        """Test that extra unknown fields are rejected"""
        with pytest.raises(ValidationError):
            ModelConfig(
                base_model="org/model",
                new_model_name="test",
                quantization="nf4",
                unknown_field="value"  # Should be rejected
            )
