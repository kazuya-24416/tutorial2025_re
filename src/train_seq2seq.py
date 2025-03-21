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
from utils import formatting_prompts_func, preprocess_logits_for_metrics, read_jsonlines

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Extract configuration values
model_name_or_path = config["model_name_or_path"]
dataset_path = config["dataset_path"]
response_template = config["response_template"]
use_lora = config.get("use_lora", False)
use_quantization = config.get("use_quantization", False)

# Load dataset
dataset = read_jsonlines(dataset_path)
dataset = Dataset.from_pandas(dataset)

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
def preprocess_function(examples):
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
        instruction_text = f"### Instruction: 以下の文章から関係トリプルを抽出してください。\n        関係トリプルは(エンティティ1, 関係, エンティティ2)の形式で出力してください。\n        {examples['instruction'][i]}\n\n### Response:\n"
        instructions.append(instruction_text)

    # Store the instruction-only text for generation during evaluation
    model_inputs["instruction_text"] = instructions

    return model_inputs


# Apply preprocessing to dataset
processed_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset.column_names
)


# Define a custom data collator that masks the instruction part with -100
class CustomDataCollatorForSeq2Seq:
    def __init__(
        self, tokenizer, model, response_template, padding="max_length", max_length=512
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.response_template = response_template
        # Tokenize the response template to find its start in the sequence
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, features):
        # Tokenizer's default collation
        batch = {}

        # Handling input_ids and attention_mask
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]

        # Pad input_ids and attention_mask
        if self.padding == "max_length":
            input_ids = [
                ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
                if len(ids) < self.max_length
                else ids[: self.max_length]
                for ids in input_ids
            ]
            attention_mask = [
                mask + [0] * (self.max_length - len(mask))
                if len(mask) < self.max_length
                else mask[: self.max_length]
                for mask in attention_mask
            ]

        batch["input_ids"] = torch.tensor(input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)

        # Create labels with -100 for instruction part
        labels = []
        for feature_input_ids in input_ids:
            label = feature_input_ids.copy()

            # Find the position of response_template in the sequence
            response_start_idx = -1
            for i in range(len(label) - len(self.response_token_ids) + 1):
                if (
                    label[i : i + len(self.response_token_ids)]
                    == self.response_token_ids
                ):
                    response_start_idx = i
                    break

            # If response template is found, mask everything before it with -100
            if response_start_idx != -1:
                label[:response_start_idx] = [-100] * response_start_idx

            labels.append(label)

        batch["labels"] = torch.tensor(labels)

        return batch


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

# Add generation-specific parameters for Seq2SeqTrainer
training_args["predict_with_generate"] = True
# Set generation parameters correctly
training_args["generation_max_length"] = 768  # Input length (512) + some extra tokens
training_args["generation_num_beams"] = 4

# Remove parameters not supported by Seq2SeqTrainingArguments
if "max_seq_length" in training_args:
    del training_args["max_seq_length"]

# Create Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(**training_args)


# Create a custom Seq2SeqTrainer that uses instruction-only inputs during evaluation
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to use instruction-only inputs during generation."""
        if self.args.predict_with_generate and not prediction_loss_only:
            # For generation, we want to use the instruction-only inputs
            # Get the batch size
            batch_size = inputs["input_ids"].shape[0]

            # Get instruction texts from the dataset
            instruction_texts = []
            for i in range(batch_size):
                # Find the index in the dataset
                idx = inputs.get("idx", torch.tensor(list(range(batch_size))))[i].item()
                if idx < len(self.eval_dataset):
                    instruction_text = self.eval_dataset[idx].get(
                        "instruction_text", ""
                    )
                    instruction_texts.append(instruction_text)
                else:
                    # Fallback if index is out of range
                    instruction_texts.append("")

            # Tokenize instruction texts
            if instruction_texts:
                instruction_inputs = self.tokenizer(
                    instruction_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(inputs["input_ids"].device)

                # Replace the input_ids with instruction-only input_ids for generation
                generation_inputs = {"input_ids": instruction_inputs["input_ids"]}
                if "attention_mask" in instruction_inputs:
                    generation_inputs["attention_mask"] = instruction_inputs[
                        "attention_mask"
                    ]

                return super().prediction_step(
                    model,
                    generation_inputs,
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                )

        # For loss calculation, use the original inputs
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )


# Create custom Seq2SeqTrainer
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    processing_class=tokenizer,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model(log_dir + "/final_model")
