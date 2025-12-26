import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config_validation import load_config

# Load and validate configuration
config = load_config("config/training_config.yaml")

BASE_MODEL = config.model.base_model
ADAPTER_PATH = config.model.new_model_name

# The "Golden Set" - Questions that separate Junior Admins from Senior Architects
TEST_QUESTIONS = [
    "Design a multi-region disaster recovery strategy for a financial app on AWS using Route 53 and Aurora Global Database.",
    "Explain the security implications of enabling 'privileged' mode in a Kubernetes container.",
    "Compare the use of Azure Service Endpoints vs. Private Link for securing SQL Database access.",
    "How does OIDC federation differ from SAML 2.0 when integrating Okta with AWS IAM?"
]

def evaluate():
    print("Loading Llama 3.1 + Adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        BASE_MODEL, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)  # nosec B615

    print("\n--- ARCHITECTURAL REVIEW ---\n")
    for question in TEST_QUESTIONS:
        # Construct Prompt (Llama 3 Format)
        messages = [
            {"role": "system", "content": "You are a Senior Cloud Security Architect."},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Q: {question}")
        print(f"A: {response}\n")
        print("-" * 60)

if __name__ == "__main__":
    evaluate()