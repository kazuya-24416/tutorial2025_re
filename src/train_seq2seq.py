from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
from compute_metrics_re import get_compute_metrics_function
from custom_collator import CustomDataCollatorForSeq2Seq
from utils import formatting_prompts_func, preprocess_logits_for_metrics, read_jsonlines

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

# Load dataset
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

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,  # Use float16 for better performance
    device_map="auto",  # Automatically distribute model across available GPUs
    **quantization_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Set padding side to left for decoder-only models (as per warning)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Apply LoRA if enabled
lora_config = None
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
        instruction_text = f"### Instruction: 以下の文章から関係トリプルを抽出してください。\n関係トリプルは(エンティティ1, 関係, エンティティ2)の形式で出力してください。\n{examples['instruction'][i]}\n\n### Response:\n"
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

# Set up logging directory
hprams_name = f"""lr{config["training_args"]["learning_rate"]}_bs{config["training_args"]["per_device_train_batch_size"]}"""
log_dir = config["training_args"]["output_dir"] + f"/{hprams_name}"
Path(log_dir).mkdir(parents=True, exist_ok=True)
compute_metrics = get_compute_metrics_function(tokenizer, log_dir)

# Configure Weights & Biases
wandb_config = config.get("wandb", {})
wandb_enabled = wandb_config.get("enabled", True)

# Update report_to based on wandb configuration
training_args = config["training_args"].copy()
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

# Create Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(**training_args)

# Create custom Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# Start training
trainer.train()
