import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

# Set environment variables to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Custom model configuration
config = GPT2Config(
    _name_or_path='gpt_test',
    vocab_size=50257,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=4,
    n_head=4,
    activation_function='gelu_new',
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-05,
    initializer_range=0.02,
    summary_type='cls_index',
    summary_use_proj=True,
    summary_activation=None,
    summary_proj_to_labels=True,
    summary_first_dropout=0.1,
    bos_token_id=50256,
    eos_token_id=50256,
)

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Print model configuration to ensure parameters are set correctly
print("Model configuration:", config)

# Load WikiText-2 dataset
datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, padding='max_length', truncation=True, max_length=512)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

# Print a sample of the tokenized dataset to ensure proper tokenization
print("Tokenized dataset sample:", tokenized_datasets["train"][0])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,  # Enable mixed precision training
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    report_to='none'  # Disable reporting to reduce unnecessary output
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Print training arguments to ensure parameters are set correctly
print("Training arguments:", training_args)

# Start training
try:
    trainer.train()
except Exception as e:
    print(f"Training failed with error: {e}")

# Save model and tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
