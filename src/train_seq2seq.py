from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
from compute_metrics_re import get_compute_metrics
from custom_collator import CustomDataCollatorForSeq2Seq
from utils import preprocess_function_eval, preprocess_function_train, read_jsonlines

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Setup quantization configuration if enabled
quantization_kwargs = {}
bnb_config = BitsAndBytesConfig(**config["quantization_config"])
quantization_kwargs["quantization_config"] = bnb_config

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config["model_name_or_path"],
    torch_dtype=torch.float16,  # Use float16 for better performance
    device_map="auto",  # Automatically distribute model across available GPUs
    **quantization_kwargs,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
# Set padding side to left for decoder-only models (as per warning)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Configure LoRA
lora_config = LoraConfig(**config["lora_config"])
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters info

# Load dataset
train_dataset = read_jsonlines(config["train_dataset_path"])
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = read_jsonlines(config["eval_dataset_path"])
eval_dataset = Dataset.from_pandas(eval_dataset)
# Apply preprocessing to dataset
train_dataset = train_dataset.map(
    preprocess_function_train, batched=True, remove_columns=train_dataset.column_names
)
eval_dataset = eval_dataset.map(
    preprocess_function_eval, batched=True, remove_columns=eval_dataset.column_names
)

# Create custom data collator that masks instruction part
data_collator = CustomDataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    response_template=config["response_template"],
    padding="max_length",
    max_length=config["max_length"],
)

# Weights & Biases
wandb.init(**config["wandb"])

# 評価スクリプトを呼び出して下さい
compute_metrics = get_compute_metrics(tokenizer, config["training_args"]["output_dir"])

# Create Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(**config["training_args"])
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# Add early stopping callback if enabled
trainer.add_callback(EarlyStoppingCallback(**config["early_stopping"]))

# Start training
trainer.train()
