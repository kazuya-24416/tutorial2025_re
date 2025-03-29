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
from compute_metrics_re import compute_metrics
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

# Get hyperparameter search parameters from config (default values if not specified)
hp_search_enabled = config.get("hyperparameter_search", {}).get("enabled", False)
hp_search_n_trials = config.get("hyperparameter_search", {}).get("n_trials", 10)
hp_search_backend = config.get("hyperparameter_search", {}).get("backend", "wandb")
hp_search_compute_objective = config.get("hyperparameter_search", {}).get(
    "compute_objective", None
)
hp_search_direction = config.get("hyperparameter_search", {}).get(
    "direction", "maximize"
)


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
            f"{examples['instruction'][i]}\n\n{config['response_template']}"
        )
        instructions.append(instruction_text)

    # Store the instruction-only text for generation during evaluation
    model_inputs["instruction_text"] = instructions

    return model_inputs


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


# モデル初期化関数
def model_init() -> AutoModelForCausalLM:
    """Initialize model for hyperparameter search.

    Returns:
        AutoModelForCausalLM: The initialized model.

    """
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

    # Load model
    initialized_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,  # Use float16 for better performance
        device_map="auto",  # Automatically distribute model across available GPUs
        **quantization_kwargs,
    )

    # Apply LoRA if enabled
    if use_lora:
        # Prepare model for k-bit training if quantization is enabled
        if use_quantization:
            initialized_model = prepare_model_for_kbit_training(initialized_model)

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
        initialized_model = get_peft_model(initialized_model, lora_config)
        initialized_model.print_trainable_parameters()

    return initialized_model


# ハイパーパラメータ探索空間の定義
def wandb_hp_space(trial: dict) -> dict:  # noqa: ARG001
    """Define hyperparameter search space for wandb.

    Args:
        trial: Trial object.

    Returns:
        dict: Hyperparameter search space.

    """
    # HP探索のパラメータをconfigから取得
    hp_config = config.get("hyperparameter_search", {}).get("parameters", {})

    # 学習率の探索範囲
    lr_min = hp_config.get("learning_rate_min", 1e-6)
    lr_max = hp_config.get("learning_rate_max", 1e-4)

    # バッチサイズの探索値
    batch_sizes = hp_config.get("batch_sizes", [16, 32, 64, 128])

    # 探索方法
    search_method = hp_config.get("method", "random")

    # 評価指標
    metric_name = hp_config.get("metric_name", "eval_f1")
    metric_goal = hp_config.get("metric_goal", "maximize")

    return {
        "method": search_method,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": lr_min,
                "max": lr_max,
            },
            "per_device_train_batch_size": {"values": batch_sizes},
        },
    }


def compute_objective(metrics: dict) -> float:
    """Compute objective for hyperparameter search.

    Args:
        metrics (dict): Metrics from evaluation.

    Returns:
        float: Objective value.

    """
    # 設定から評価指標を取得、デフォルトはeval_f1
    metric_name = config.get("hyperparameter_search", {}).get("metric_name", "eval_f1")
    return metrics.get(metric_name)


# トークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Set padding side to left for decoder-only models (as per warning)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


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
    model=None,
    response_template=response_template,
    padding="max_length",
    max_length=512,
)

# wandbの設定部分を修正
wandb_config = config.get("wandb", {})
wandb_enabled = wandb_config.get("enabled", True)
wandb_project = wandb_config.get("project", "tutorial2025_re")

# Update report_to based on wandb configuration
training_args = config["training_args"].copy()
if not wandb_enabled:
    # Disable wandb
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
elif not hp_search_enabled:
    # hyperparameter searchを使わない場合のみinitする
    wandb.init(
        project=wandb_project,
        name=wandb_config.get("name", None),
        group=wandb_config.get("group", None),
        tags=wandb_config.get("tags", []),
        mode="disabled" if wandb_config.get("debug", False) else "online",
    )

# Set up logging directory
base_output_dir = config["training_args"]["output_dir"]
Path(base_output_dir).mkdir(parents=True, exist_ok=True)

# ハイパーパラメータ探索用のTrainingArgsを作成
# LRとバッチサイズはハイパーパラメータ探索で上書きされるため固定値を設定
args = Seq2SeqTrainingArguments(
    output_dir=base_output_dir,
    evaluation_strategy=training_args.get("evaluation_strategy", "epoch"),
    save_strategy=training_args.get("save_strategy", "epoch"),
    logging_strategy=training_args.get("logging_strategy", "steps"),
    logging_steps=training_args.get("logging_steps", 500),
    num_train_epochs=training_args.get("num_train_epochs", 3),
    save_total_limit=training_args.get("save_total_limit", 3),
    load_best_model_at_end=training_args.get("load_best_model_at_end", True),
    metric_for_best_model=training_args.get("metric_for_best_model", "eval_f1"),
    greater_is_better=training_args.get("greater_is_better", True),
    report_to=training_args.get("report_to", "wandb"),
    # ハイパーパラメータ探索で探索するパラメータなので固定値を設定
    learning_rate=training_args.get("learning_rate", 2e-5),
    per_device_train_batch_size=training_args.get("per_device_train_batch_size", 16),
    per_device_eval_batch_size=training_args.get("per_device_eval_batch_size", 16),
    # その他のパラメータ
    weight_decay=training_args.get("weight_decay", 0.01),
    adam_beta1=training_args.get("adam_beta1", 0.9),
    adam_beta2=training_args.get("adam_beta2", 0.999),
    adam_epsilon=training_args.get("adam_epsilon", 1e-8),
    max_grad_norm=training_args.get("max_grad_norm", 1.0),
    predict_with_generate=training_args.get("predict_with_generate", True),
    generation_max_length=training_args.get("generation_max_length", 512),
)

# Create custom Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=None,  # モデルはハイパーパラメータ探索中にmodel_init関数で初期化
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=create_compute_metrics_function(tokenizer, base_output_dir),
    tokenizer=tokenizer,
    model_init=model_init,  # ハイパーパラメータ探索用のモデル初期化関数
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

# ハイパーパラメータ探索を実行
if hp_search_enabled:
    best_run = trainer.hyperparameter_search(
        hp_space=wandb_hp_space,
        n_trials=hp_search_n_trials,
        direction=hp_search_direction,
        backend=hp_search_backend,
        compute_objective=compute_objective,
    )

    # 最適なハイパーパラメータで再設定
    for param, value in best_run.hyperparameters.items():
        setattr(trainer.args, param, value)

    # 最適なハイパーパラメータでモデルを再初期化
    model = model_init()
    trainer.model = model

    # 最適なハイパーパラメータでログディレクトリを更新
    hprams_name = f"""lr{best_run.hyperparameters["learning_rate"]}_bs{best_run.hyperparameters["per_device_train_batch_size"]}"""  # noqa: E501
    log_dir = base_output_dir + f"/{hprams_name}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    trainer.args.output_dir = log_dir

    # 最適なハイパーパラメータで通常のトレーニングを実行
    trainer.train()
else:
    # 通常のトレーニングを実行
    model = model_init()
    trainer.model = model

    # 通常のログディレクトリを設定
    hprams_name = f"""lr{training_args["learning_rate"]}_bs{training_args["per_device_train_batch_size"]}"""  # noqa: E501
    log_dir = base_output_dir + f"/{hprams_name}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    trainer.args.output_dir = log_dir

    trainer.train()
