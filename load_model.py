from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token="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")

# Specify the model name
#model_name = "facebook/opt-1.3b"  # You can change this to the desired model size
model_name = "mistralai/Mistral-7B-v0.1"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)

print("Model and tokenizer loaded successfully!")