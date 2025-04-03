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
# 課題44：configからハイパーパラメータを読み込みなさい

# Setup quantization configuration if enabled
quantization_kwargs = {}
bnb_config = # 課題39：量子化の設定を記入
quantization_kwargs["quantization_config"] = bnb_config

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    # 課題39：モデルの指定や量子化の設定を記入
)

# Configure LoRA
lora_config = # 課題39：LoRAの設定を記入
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters info

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained() # モデルの指定
# Set padding side to left for decoder-only models (as per warning)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_dataset = # 課題40：訓練データの読み込み
eval_dataset = # 課題40：評価データの読み込み
# Apply preprocessing to dataset
train_dataset = train_dataset.map(
    # 課題40：前処理を適用
)
eval_dataset = eval_dataset.map(
    # 課題40：前処理を適用
)

# data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
    max_length=,
    label_pad_token_id=-100,
)

# Weights & Biases
wandb.init(
    # 課題41：Weights & Biasesの設定を記入
)

# compute metrics
compute_metrics = # 課題43：評価スクリプトを呼び出して下さい

# Create Seq2SeqTrainingArguments
args = # 課題42：Trainerの設定を記入
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
    # 課題43：Early Stoppingの設定を記入
)

# Start training
trainer.train()
