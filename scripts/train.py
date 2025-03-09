import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk

# Load Dataset
dataset = load_from_disk("data/processed/hf_dataset")

# Load T5-small Model & Tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to MPS (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Tokenize Inputs with Padding and Truncation
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=512,
        padding="max_length",  # Ensure consistent length
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        examples["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply Tokenization
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Data Collator for Padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce batch size for MPS
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
    fp16=False,  # MPS does not support fp16
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the Model
trainer.train()

# Save Model
trainer.save_model("models/t5_summarization")
print("✅ Model trained and saved at models/t5_summarization")
