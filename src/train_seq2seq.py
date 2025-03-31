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
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

import wandb
from compute_metrics_re import compute_metrics
from custom_collator import CustomDataCollatorForSeq2Seq
from utils import formatting_prompts_func, read_jsonlines

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Load dataset
train_dataset = read_jsonlines(train_dataset_path)
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = read_jsonlines(eval_dataset_path)
eval_dataset = Dataset.from_pandas(eval_dataset)


# Setup quantization configuration if enabled
quantization_kwargs = {}
bnb_config = BitsAndBytesConfig(
    # 量子化の設定を指定して下さい
)
quantization_kwargs["quantization_config"] = bnb_config

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "", # モデル名を指定して下さい
    torch_dtype=torch.float16,  # Use float16 for better performance
    device_map="auto",  # Automatically distribute model across available GPUs
    **quantization_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained("") # モデル名を指定して下さい

# Set padding side to left for decoder-only models (as per warning)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



# Configure LoRA
lora_config = LoraConfig(
    # LoRAの設定を指定して下さい
)
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters info


# Process dataset to create input_ids and labels
def preprocess_function(examples: dict) -> dict:
    """Preprocess the dataset to create input_ids and labels.

    Args:
        examples (dict): A dictionary containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed dataset.

    """
    formatted_texts = formatting_prompts_func(examples)

    # Tokenize the texts
    model_inputs = tokenizer(
        formatted_texts, padding="max_length", truncation=True, max_length=512
    )

    # Create attention mask
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = [
            [1] * len(input_ids) for input_ids in model_inputs["input_ids"]
        ]

    # For evaluation, we also need to create instruction-only inputs
    # Extract instruction part (everything before the response template)
    instructions = []
    for i in range(len(examples["instruction"])):
        instruction_text = f"{examples['instruction'][i]}\n\n{config['response_template']}" # noqa: E501
        instructions.append(instruction_text)

    # Store the instruction-only text for generation during evaluation
    model_inputs["instruction_text"] = instructions

    return model_inputs


# Apply preprocessing to dataset
train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names
)

# Create custom data collator that masks instruction part
data_collator = CustomDataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    response_template=response_template,
    padding="max_length",
    max_length=512,
)

# Weights & Biases
wandb.init(
    # Weights & Biasesの設定を指定して下さい
)

# 評価スクリプトを呼び出して下さい
compute_metrics=

# Create Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(
    # Trainerの設定を指定して下さい
)
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
trainer.add_callback(
    # EarlyStoppingCallbackの設定を指定して下さい
)

# Start training
trainer.train()
