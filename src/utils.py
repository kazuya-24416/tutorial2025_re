import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])


# Load dataset
def read_jsonlines(file_path: str) -> pd.DataFrame:
    """Read a jsonlines file and convert it to a pandas DataFrame.

    Args:
        file_path (str): Path to the jsonlines file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the jsonlines file.

    """
    data = []
    with Path(file_path).open(encoding="utf-8") as f:
        data.extend([json.loads(line.strip()) for line in f])
    return pd.DataFrame(data)


# Process dataset to create input_ids and labels
def preprocess_function_train(examples: dict) -> dict:
    """Preprocess the dataset to create input_ids and labels.

    Args:
        examples (dict): A dictionary containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed dataset.

    """
    concat_texts = []
    for i in range(len(examples["instruction"])):
        # Format the prompt in a way that's consistent with the model's expected format
        text = f"{examples['instruction'][i]}\n\n{config['response_template']} {examples['output'][i]}"  # noqa: E501
        concat_texts.append(text)

    # Tokenize the texts
    model_inputs = tokenizer(
        concat_texts, padding="max_length", truncation=True, max_length=512
    )

    # Create attention mask
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = [
            [1] * len(input_ids) for input_ids in model_inputs["input_ids"]
        ]

    return model_inputs


def preprocess_function_eval(examples: dict) -> dict:
    """Preprocess the dataset to create input_ids and labels for evaluation.

    Args:
        examples (dict): A dictionary containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed dataset with input_ids and labels.

    """
    # 入力部分のみ（指示部分とレスポンステンプレート）
    input_texts = []
    # 全体（指示部分と出力部分）- ラベル用
    full_texts = []

    for i in range(len(examples["instruction"])):
        # 入力部分（モデルに渡す）
        instruction_text = (
            f"{examples['instruction'][i]}\n\n{config['response_template']}"
        )
        input_texts.append(instruction_text)

        # 全体（ラベルとして使用）
        full_text = f"{examples['instruction'][i]}\n\n{config['response_template']} {examples['output'][i]}"
        full_texts.append(full_text)

    # 入力テキストをトークン化（モデルへの入力用）
    model_inputs = tokenizer(
        input_texts, padding="max_length", truncation=True, max_length=512
    )

    # 全体をトークン化（ラベル用）
    labels = tokenizer(
        full_texts, padding="max_length", truncation=True, max_length=512
    ).input_ids

    # labelsを設定
    model_inputs["labels"] = labels

    # アテンションマスクの作成
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = [
            [1] * len(input_ids) for input_ids in model_inputs["input_ids"]
        ]

    return model_inputs


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    """Original Trainer may have a memory leak.

    Args:
        logits (torch.Tensor): Logits from the model.
        labels (torch.Tensor): True labels.

    Returns:
        torch.Tensor: Predicted IDs.

    """
    return torch.argmax(logits, dim=-1)
