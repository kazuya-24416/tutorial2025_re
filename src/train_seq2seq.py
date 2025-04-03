from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
from compute_metrics_re import get_compute_metrics
from utils import preprocess_function_eval, preprocess_function_train, read_jsonlines

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Setup quantization configuration if enabled
quantization_kwargs = {}
bnb_config = # 量子化の設定を記入
quantization_kwargs["quantization_config"] = bnb_config

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    # モデルの指定や量子化の設定を記入
)

# Configure LoRA
lora_config = # LoRAの設定を記入
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters info

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
# Set padding side to left for decoder-only models (as per warning)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_dataset = # 訓練データの読み込み
train_dataset = # データセットクラスに変換
eval_dataset = # 評価データの読み込み
eval_dataset = # データセットクラスに変換
# Apply preprocessing to dataset
train_dataset = train_dataset.map(
    # 前処理を適用
)
eval_dataset = eval_dataset.map(
    # 前処理を適用
)

# Create custom data collator that masks instruction part
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
    max_length=config["max_length"],
    label_pad_token_id=-100,
)

# Weights & Biases
wandb.init(
    # Weights & Biasesの設定を記入
)

# compute metrics
compute_metrics = # 評価スクリプトを呼び出して下さい

# Create Seq2SeqTrainingArguments
args = # Trainerの設定を記入
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Early Stopping
trainer.add_callback(
    # Early Stoppingの設定を記入
)

# Start training
trainer.train()
