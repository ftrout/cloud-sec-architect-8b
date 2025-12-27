#!/usr/bin/env python3
"""
Inference script for cloud-sec-architect-8b.

Simple command-line interface for testing the model.

Usage:
    python scripts/inference.py "What are the key security considerations for AWS Lambda?"
    python scripts/inference.py --interactive
    python scripts/inference.py --hf-model fmt0816/cloud-sec-architect-8b "Your question"
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Default configuration
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER_PATH = "./cloud-sec-architect-8b"
SYSTEM_PROMPT = """You are a Senior Cloud Security Architect. Provide detailed, secure, and compliant technical guidance. Your expertise includes:
- Multi-cloud security architecture (AWS, Azure, GCP)
- Compliance frameworks (NIST 800-53, CIS Benchmarks, ISO 27001)
- Identity and access management (OIDC, SAML, OAuth 2.0)
- Infrastructure-as-Code security (Terraform, CloudFormation)
- Kubernetes and container security
- Zero Trust architecture principles"""


class CloudSecurityArchitect:
    """Wrapper for cloud-sec-architect-8b model."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
    ):
        """Initialize the model."""
        print(f"Loading base model: {base_model_id}")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load adapter
        print(f"Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)  # nosec B615
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully!\n")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response to the prompt."""
        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        return response.strip()


def interactive_mode(model: CloudSecurityArchitect, args):
    """Run interactive chat mode."""
    print("=" * 60)
    print("Cloud Security Architect AI - Interactive Mode")
    print("=" * 60)
    print("Type your questions about cloud security architecture.")
    print("Commands: 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if prompt.lower() == "clear":
            print("\n" * 50)
            print("Conversation cleared.")
            continue

        print("\nArchitect: ", end="", flush=True)

        try:
            response = model.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)
        except Exception as e:
            print(f"\nError generating response: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for cloud-sec-architect-8b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="The prompt to send to the model",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"Path to LoRA adapter (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (overrides --adapter-path)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9)",
    )

    args = parser.parse_args()

    # Validate args
    if not args.prompt and not args.interactive:
        parser.error("Either provide a prompt or use --interactive mode")

    # Determine adapter path
    adapter_path = args.hf_model if args.hf_model else args.adapter_path

    # Check if adapter exists (for local paths)
    if not args.hf_model and not os.path.exists(adapter_path):
        print(f"Error: Adapter not found at {adapter_path}")
        print("\nOptions:")
        print("  1. Train the model first: python train.py")
        print("  2. Specify a different path: --adapter-path /path/to/adapter")
        print("  3. Use HuggingFace Hub: --hf-model fmt0816/cloud-sec-architect-8b")
        return 1

    # Load model
    try:
        model = CloudSecurityArchitect(args.base_model, adapter_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    if args.interactive:
        interactive_mode(model, args)
    else:
        response = model.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
