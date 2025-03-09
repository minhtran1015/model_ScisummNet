import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load Trained Model
model_name = "models/t5_summarization"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Function to generate a summary
def summarize(text):
    inputs = tokenizer("Summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to MPS
    summary_ids = model.generate(**inputs, max_length=128)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test on an example
test_text = "This paper explores the impact of new algorithms on machine translation. The results show significant improvements."
print("Generated Summary:", summarize(test_text))
