from pathlib import Path

import torch
import yaml
from datasets import Dataset
from huggingface_hub import login
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
from compute_metrics_genia11_ner import compute_metrics
from custom_collator import CustomDataCollatorForSeq2Seq
from utils import formatting_prompts_func, read_jsonlines

login("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")  # トークンを設定

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

# Get early stopping parameters from config
early_stopping_enabled = config.get("early_stopping", {}).get("enabled", False)
early_stopping_patience = config.get("early_stopping", {}).get("patience", 3)
early_stopping_threshold = config.get("early_stopping", {}).get("threshold", 0.0)
early_stopping_metric = config.get("early_stopping", {}).get("metric", "eval_f1")

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
        instruction_text = (
            f"{examples['instruction'][i]}\n\n{config['response_template']}"  # noqa: E501
        )
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


# カスタムコールバックでエポック情報を取得
class EpochLoggerCallback(TrainerCallback):
    """Custom callback to log epoch information."""

    def __init__(self, trainer: Seq2SeqTrainer) -> None:
        """Init.

        Args:
        trainer (Seq2SeqTrainer): Trainer instance.

        """
        self.trainer = trainer

    def on_evaluate(
        self,
        args: Seq2SeqTrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        """on_evaluate callback.

        Args:
        args (Seq2SeqTrainingArguments): Training arguments.
        state (TrainerState): Trainer state.
        control (TrainerControl): Trainer control.
        **kwargs: Additional keyword arguments.

        """
        # エポック情報を更新
        self.trainer.compute_metrics_epoch = state.epoch


# Set up logging directory
hprams_name = f"""lr{config["training_args"]["learning_rate"]}_bs{config["training_args"]["per_device_train_batch_size"]}"""  # noqa: E501
log_dir = config["training_args"]["output_dir"] + f"/{hprams_name}"
Path(log_dir).mkdir(parents=True, exist_ok=True)


def create_compute_metrics_function(tokenizer: AutoTokenizer, log_dir: str) -> callable:  # noqa: ARG001
    """Create compute metrics function.

    Args:
        tokenizer (AutoTokenizer): Tokenizer for decoding.
        log_dir (str): Directory to save predictions and references.
        callback (TrainerCallback): Callback to get epoch information.

    Returns:
        callable: Compute metrics function.

    """

    def call_compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        epoch = getattr(trainer, "compute_metrics_epoch", 0)
        return compute_metrics(eval_pred, log_dir, int(epoch) + 1)

    return call_compute_metrics


# Create Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(**training_args)

# Create custom Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=create_compute_metrics_function(tokenizer, log_dir),
    tokenizer=tokenizer,
)

# Add custom callback
trainer.add_callback(EpochLoggerCallback(trainer))
# Add early stopping callback if enabled
if early_stopping_enabled:
    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
    )

# Start training
trainer.train()
