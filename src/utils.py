import json
from pathlib import Path

import pandas as pd
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


def preprocess_function_train(examples: dict) -> dict:
    """Preprocess the dataset to create input_ids and labels for training.

    Args:
        examples (dict): A dictionary containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed dataset.

    """
    processed_data = {"input_ids": [], "attention_mask": [], "labels": []}

    for instruction, output in zip(
        examples["instruction"], examples["output"], strict=False
    ):
        prompt = f"{instruction}{config['response_template']}"
        full_text = prompt + output + tokenizer.eos_token

        # 全体をトークナイズ
        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=config["max_length"],
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        # 指示部分のトークナイズ
        tokenized_prompt = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(tokenized_prompt["input_ids"])

        prompt_len = min(prompt_len, len(input_ids))

        # ラベルの作成
        labels = input_ids.copy()  # リストのコピー
        labels[:prompt_len] = [-100] * prompt_len

        processed_data["input_ids"].append(input_ids)
        processed_data["attention_mask"].append(attention_mask)
        processed_data["labels"].append(labels)
        assert len(input_ids) == len(attention_mask) == len(labels), (
            len(input_ids),
            len(attention_mask),
            len(labels),
        )

    return processed_data


def preprocess_function_eval(examples: dict) -> dict:
    """Preprocess the dataset to create input_ids and labels for evaluation.

    Args:
        examples (dict): A dictionary containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed dataset.

    """
    processed_data = {"input_ids": [], "attention_mask": [], "labels": []}

    for instruction in examples["instruction"]:
        prompt = f"{instruction}{config['response_template']}"
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        attention_mask = tokenizer(prompt, add_special_tokens=False)["attention_mask"]
        labels = input_ids.copy()

        processed_data["input_ids"].append(input_ids)
        processed_data["attention_mask"].append(attention_mask)
        processed_data["labels"].append(labels)

    return processed_data
