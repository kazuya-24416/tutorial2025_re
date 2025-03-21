from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig

import wandb
from compute_metrics_re import get_compute_metrics_function
from custom_trainer import CustomSFTTrainer, GenerateEvaluationCallback
from utils import formatting_prompts_func, read_jsonlines

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Extract configuration values
model_name_or_path = config["model_name_or_path"]
train_dataset_path = config["train_dataset_path"]
eval_dataset_path = config["eval_dataset_path"]
response_template = config["response_template"]
use_lora = config.get("use_lora", False)
use_quantization = config.get("use_quantization", False)

train_dataset = read_jsonlines(train_dataset_path)
train_dataset = Dataset.from_pandas(train_dataset)

eval_dataset = read_jsonlines(eval_dataset_path)
eval_dataset = Dataset.from_pandas(eval_dataset)

# Setup quantization configuration if enabled
quantization_kwargs = {}
if use_quantization:
    quant_config = config["quantization_config"]
    bits = quant_config.get("bits", 8)  # Default to 8-bit if not specified

    # Configure quantization based on bits
    if bits == 4:
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=getattr(
                torch, quant_config.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_use_double_quant=quant_config.get(
                "bnb_4bit_use_double_quant", True
            ),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        )
    else:  # 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            llm_int8_threshold=quant_config.get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=quant_config.get(
                "llm_int8_has_fp16_weight", False
            ),
        )

    quantization_kwargs["quantization_config"] = bnb_config

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,  # Use float16 for better performance
    device_map="auto",  # Automatically distribute model across available GPUs
    **quantization_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Apply LoRA if enabled
if use_lora:
    # Prepare model for k-bit training if quantization is enabled
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora_config"].get("r", 16),
        lora_alpha=config["lora_config"].get("lora_alpha", 32),
        lora_dropout=config["lora_config"].get("lora_dropout", 0.05),
        bias=config["lora_config"].get("bias", "none"),
        task_type=config["lora_config"].get("task_type", "CAUSAL_LM"),
        target_modules=config["lora_config"].get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters info

# Create data collator with response template from config
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# eval
hprams_name = f"lr{config['training_args']['learning_rate']}_bs{config['training_args']['per_device_train_batch_size']}"
log_dir = config["training_args"]["output_dir"] + f"/{hprams_name}"
Path(log_dir).mkdir(parents=True, exist_ok=True)
compute_metrics = get_compute_metrics_function(tokenizer, log_dir)

# Get training arguments from config
training_args = config["training_args"]

# Configure Weights & Biases
wandb_config = config.get("wandb", {})
wandb_enabled = wandb_config.get("enabled", True)

# Update report_to based on wandb configuration
if not wandb_enabled:
    # Disable wandb during debugging
    if "report_to" in training_args and "wandb" in training_args["report_to"]:
        if training_args["report_to"] == "wandb":
            training_args["report_to"] = "none"
        elif training_args["report_to"] == "all":
            training_args["report_to"] = ["tensorboard"]
        elif (
            isinstance(training_args["report_to"], list)
            and "wandb" in training_args["report_to"]
        ):
            training_args["report_to"] = [
                r for r in training_args["report_to"] if r != "wandb"
            ]
else:
    # Initialize wandb with project settings
    wandb.init(
        project=wandb_config.get("project", "tutorial2025_re"),
        name=wandb_config.get("name", None),
        group=wandb_config.get("group", None),
        tags=wandb_config.get("tags", []),
        mode="disabled" if wandb_config.get("debug", False) else "online",
    )


# Create SFTConfig from training arguments
args = SFTConfig(**training_args)

# Create SFT trainer
trainer = CustomSFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=lora_config if use_lora else None,
)

# Add custom callback
trainer.add_callback(
    GenerateEvaluationCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics_fn=compute_metrics,
        generate_kwargs={"max_new_tokens": 128, "num_beams": 1},
    )
)


# Start training
trainer.train()
