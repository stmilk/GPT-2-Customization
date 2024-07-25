import torch
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, EvalPrediction
from datasets import load_from_disk, load_metric
import sys
import os
import numpy as np

# Import custom modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenization'))
from token_processor import CustomChineseTokenizer, load_vocab_from_txt

# Set environment variables to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

token_table = load_vocab_from_txt()
tokenizer = CustomChineseTokenizer(token_table)

# Custom model configuration
config = GPT2Config(
    _name_or_path='gpt_test',
    vocab_size=tokenizer.get_vocab_num(),
    n_positions=512,
    n_ctx=512,
    n_embd=128,
    n_layer=3,
    n_head=8,
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
    bos_token_id=tokenizer.get_bos_token(),
    eos_token_id=tokenizer.get_eos_token()
)

# Load GPT-2 model
model = GPT2LMHeadModel(config)

# Custom data collator
def custom_data_collator(features):
    batch = {
        'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
        'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
        'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)
    }
    return batch

# Load preprocessed datasets
tokenized_datasets = load_from_disk(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/data_preprocess"))

# Print a sample of the tokenized dataset to ensure proper tokenization
print("Tokenized dataset sample:", tokenized_datasets["train"][0])

# Load metrics
accuracy_metric = load_metric("accuracy", trust_remote_code=True)
perplexity_metric = load_metric("perplexity", trust_remote_code=True)

# Define compute metrics function
def compute_metrics(p: EvalPrediction):
    logits = p.predictions
    labels = p.label_ids

    # Ignore padding tokens
    mask = labels != tokenizer.pad_token_id

    # Compute accuracy
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions[mask], references=labels[mask])["accuracy"]

    # Compute perplexity
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    perplexity = np.exp(perplexity_metric.compute(predictions=shift_logits, references=shift_labels)["perplexity"])

    return {"accuracy": accuracy, "perplexity": perplexity}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=4,
    fp16=True,  # Enable mixed precision training
    save_steps=2000,
    save_total_limit=2,
    learning_rate=1e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    report_to='tensorboard',  # Enable reporting to TensorBoard
    logging_dir='./logs',  # Directory for TensorBoard logs
    logging_steps=500,  # Log every 500 steps
    max_grad_norm=1.0  # Add gradient clipping
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics  # Add compute metrics function
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

# Clear cache before evaluation
torch.cuda.empty_cache()

# Evaluate model performance using CPU
eval_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=2,  # Smaller batch size for evaluation
    no_cuda=False  # Use CPU for evaluation
)
eval_trainer = Trainer(
    model=model,
    args=eval_args,
    data_collator=custom_data_collator,
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

try:
    evaluation_results = eval_trainer.evaluate()
    print(f"Evaluation results: {evaluation_results}")
except RuntimeError as e:
    print(f"Evaluation failed with error: {e}")

# Start TensorBoard
print("Start TensorBoard with: tensorboard --logdir=./logs")
