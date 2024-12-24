import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Set the pad_token explicitly to eos_token
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model uses the pad token

# Prepare input with attention mask
sentence = 'Today is a good day'
inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

# Move input to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}  # Move all inputs to device

# Generate text with attention mask
result = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],  # Explicit attention mask
    max_length=50,
    num_beams=2,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Decode and print result
generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
print(generated_text)

