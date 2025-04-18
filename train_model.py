import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ✅ Disable WandB (Prevents prompts)
os.environ["WANDB_DISABLED"] = "true"

# ✅ Load GPT-2 Model & Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ✅ Fix Padding Token Issue
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

# ✅ Load Custom Dataset
def load_data():
    with open("/content/dataset.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    return Dataset.from_dict({"text": lines})

dataset = load_data()

# ✅ Tokenization Function (Correct Label Formatting)
def tokenize(example):
    encoding = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    encoding["labels"] = encoding["input_ids"]  # ✅ Correct label formatting
    return encoding

dataset = dataset.map(tokenize)

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="/content/results",
    report_to="none",  # ✅ Prevents W&B logging
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs"
)

# ✅ Initialize Trainer (No CustomTrainer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ✅ Start Fine-Tuning
trainer.train()
