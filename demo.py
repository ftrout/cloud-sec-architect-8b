#!/usr/bin/env python3
"""
Gradio Demo for cloud-sec-architect-8b

A simple web interface for testing the Cloud Security Architect model.
Supports both local adapter paths and HuggingFace Hub models.

Usage:
    python demo.py                           # Use default local path
    python demo.py --adapter-path ./my-model # Custom local path
    python demo.py --hf-model ftrout/cloud-sec-architect-8b  # HuggingFace Hub
"""

import argparse
import os
from typing import Generator

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread


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


def load_model(base_model_id: str, adapter_path: str):
    """Load the base model with fine-tuned adapter."""
    print(f"Loading base model: {base_model_id}")
    print(f"Loading adapter from: {adapter_path}")

    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)  # nosec B615
    tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(
    message: str,
    history: list,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """Generate a streaming response from the model."""

    # Build conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    # Tokenize input
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    # Set up streaming
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Generation config
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Run generation in a separate thread for streaming
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the response
    partial_response = ""
    for new_text in streamer:
        partial_response += new_text
        yield partial_response

    thread.join()


def create_demo(model, tokenizer) -> gr.Blocks:
    """Create the Gradio interface."""

    # Example prompts for users
    example_prompts = [
        "What are the key security considerations when designing a multi-region AWS architecture?",
        "Compare Azure Private Endpoints vs Service Endpoints for a healthcare workload.",
        "How should I implement Zero Trust architecture for a hybrid cloud environment?",
        "What Kubernetes security controls are needed for PCI-DSS compliance?",
        "When should I use OIDC vs SAML for enterprise identity federation?",
        "Design a secure CI/CD pipeline with proper secrets management.",
        "What are the best practices for securing Terraform state files?",
        "How do I implement network segmentation in a multi-cloud environment?",
    ]

    with gr.Blocks(
        title="Cloud Security Architect AI",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 900px !important; }
        .disclaimer { font-size: 0.85em; color: #666; padding: 10px; background: #f9f9f9; border-radius: 8px; margin-top: 10px; }
        """
    ) as demo:
        gr.Markdown(
            """
            # Cloud Security Architect AI

            **cloud-sec-architect-8b** - A specialized LLM fine-tuned for cloud security architecture guidance.

            This model provides expert-level advice on:
            - Multi-cloud security architecture (AWS, Azure, GCP)
            - Compliance frameworks (NIST, CIS, ISO 27001)
            - Identity & access management
            - Infrastructure-as-Code security
            - Kubernetes & container security
            """
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=450,
            show_copy_button=True,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about cloud security architecture...",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Max New Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p (Nucleus Sampling)",
                )

        with gr.Accordion("Example Prompts", open=True):
            gr.Examples(
                examples=[[p] for p in example_prompts],
                inputs=[msg],
                label="Click an example to use it",
            )

        clear_btn = gr.Button("Clear Conversation")

        gr.Markdown(
            """
            <div class="disclaimer">
            <strong>Disclaimer:</strong> This AI model is for educational and advisory purposes only.
            Always verify recommendations against official vendor documentation and your organization's
            compliance requirements before implementation. AI models can produce inaccurate information.
            </div>
            """,
            elem_classes=["disclaimer"]
        )

        # Event handlers
        def respond(message, chat_history, max_tokens, temperature, top_p):
            """Handle user message and generate response."""
            if not message.strip():
                return "", chat_history

            # Add user message to history
            chat_history = chat_history + [[message, ""]]

            # Generate streaming response
            for partial_response in generate_response(
                message,
                chat_history[:-1],  # Exclude current empty response
                model,
                tokenizer,
                max_tokens,
                temperature,
                top_p,
            ):
                chat_history[-1][1] = partial_response
                yield "", chat_history

        def clear():
            """Clear conversation history."""
            return [], ""

        # Wire up events
        submit_btn.click(
            respond,
            [msg, chatbot, max_tokens, temperature, top_p],
            [msg, chatbot],
        )

        msg.submit(
            respond,
            [msg, chatbot, max_tokens, temperature, top_p],
            [msg, chatbot],
        )

        clear_btn.click(clear, outputs=[chatbot, msg])

    return demo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gradio Demo for cloud-sec-architect-8b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    # Determine adapter path
    adapter_path = args.hf_model if args.hf_model else args.adapter_path

    # Check if adapter exists (for local paths)
    if not args.hf_model and not os.path.exists(adapter_path):
        print(f"Error: Adapter path not found: {adapter_path}")
        print("\nOptions:")
        print("  1. Train the model first: python train.py")
        print("  2. Specify a different path: python demo.py --adapter-path /path/to/adapter")
        print("  3. Use HuggingFace Hub: python demo.py --hf-model ftrout/cloud-sec-architect-8b")
        return 1

    # Load model
    try:
        model, tokenizer = load_model(args.base_model, adapter_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Create and launch demo
    demo = create_demo(model, tokenizer)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )

    return 0


if __name__ == "__main__":
    exit(main())
