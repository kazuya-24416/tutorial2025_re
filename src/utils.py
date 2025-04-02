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


# # Process dataset to create input_ids and labels
# def preprocess_function_train(examples: dict) -> dict:
#     """Preprocess the dataset to create input_ids and labels.

#     Args:
#         examples (dict): A dictionary containing the dataset.

#     Returns:
#         dict: A dictionary containing the preprocessed dataset.

#     """
#     concat_texts = []
#     for i in range(len(examples["instruction"])):
#         # Format the prompt in a way that's consistent with the model's expected format
#         text = f"{examples['instruction'][i]}\n\n{config['response_template']} {examples['output'][i]}{tokenizer.eos_token}"  # noqa: E501
#         concat_texts.append(text)

#     # Tokenize the texts
#     model_inputs = tokenizer(
#         concat_texts, padding="max_length", truncation=True, max_length=512
#     )

#     # Create attention mask
#     if "attention_mask" not in model_inputs:
#         model_inputs["attention_mask"] = [
#             [1] * len(input_ids) for input_ids in model_inputs["input_ids"]
#         ]

#     return model_inputs
def preprocess_function_train(examples):
    processed_data = {"input_ids": [], "attention_mask": [], "labels": []}

    for instruction, output in zip(examples["instruction"], examples["output"]):
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

        # ラベルの作成
        labels = input_ids.copy()  # リストのコピー
        labels[:prompt_len] = [-100] * prompt_len

        processed_data["input_ids"].append(input_ids)
        processed_data["attention_mask"].append(attention_mask)
        processed_data["labels"].append(labels)

    return processed_data

def preprocess_function_eval(examples):
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

# def preprocess_function_eval(examples: dict) -> dict:
#     """Preprocess the dataset to create input_ids and labels for evaluation.

#     Args:
#         examples (dict): A dictionary containing the dataset.

#     Returns:
#         dict: A dictionary containing the preprocessed dataset with input_ids and labels

#     """
#     # 指示部分
#     input_texts = []
#     # 全体
#     full_texts = []

#     for i in range(len(examples["instruction"])):
#         # 指示部分
#         instruction_text = (
#             f"{examples['instruction'][i]}\n\n{config['response_template']}"
#         )
#         input_texts.append(instruction_text)

#         # 完全な参照テキスト
#         full_text = f"{examples['instruction'][i]}\n\n{config['response_template']} {examples['output'][i]}"  # noqa: E501
#         full_texts.append(full_text)

#     # 入力テキストをトークン化
#     model_inputs = tokenizer(
#         input_texts, padding="max_length", truncation=True, max_length=512
#     )

#     # 完全なテキストをトークン化
#     full_inputs = tokenizer(
#         full_texts, padding="max_length", truncation=True, max_length=512
#     )

#     # labelsを設定
#     model_inputs["labels"] = full_inputs["input_ids"]

#     # アテンションマスクの作成
#     if "attention_mask" not in model_inputs:
#         model_inputs["attention_mask"] = [
#             [1] * len(input_ids) for input_ids in model_inputs["input_ids"]
#         ]

#     return model_inputs
