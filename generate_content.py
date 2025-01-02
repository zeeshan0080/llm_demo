from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def format_seconds_to_minutes(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minutes and {remaining_seconds} seconds"


print("************ Starting Model ************")
#device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
#model.to(device)
print("-> Using:", device)
# Load the tokenizer and model
#model_name = "facebook/opt-1.3b"  # Change if necessary
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Explicitly set the pad_token_id if it is not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("-> Model loaded...")

start_time = time.time()

# Define the prompt
#prompt = "Project Proposal: Implementing a new feature for our product"
prompt = "project proposal for an ecommmerce mobile application"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate attention mask
attention_mask = inputs['input_ids'].ne(tokenizer.pad_token_id).to(device)

print("-> Inputs initialized...")
print("-> Generating output...")
# Generate text
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=attention_mask,
    #max_length=20, # 2min 29sec
    #max_length=40, # 11min 50sec
    #max_length=60,  # 16min 21 sec
    #max_length=80,  #13min 20 sec
    #max_length=100,  #30mins 19 secs
    #max_length=200, #28min 8secs
    #max_length=300, # 23mins 48secs
    max_length=15, #4mins 32secs
    #max_length=50, #104mins 31secs
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    #do_sample=True,
    #temperature=0.1,
    #top_p=0.2,
    early_stopping=False
)
print("-> Decoding output...")

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = format_seconds_to_minutes(elapsed_time)
print(formatted_time)
print("-> Output:")
print(generated_text)
