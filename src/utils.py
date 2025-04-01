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
        text = f"{examples['instruction'][i]}\n\n{config['response_template']}{examples['output'][i]}{tokenizer.eos_token}"  # noqa: E501
        concat_texts.append(text)

    # Tokenize the texts
    model_inputs = tokenizer(
        concat_texts,
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )

    # response_template以前を-100でマスクしたlabelsを設定
    labels = model_inputs["input_ids"].copy()

    # response_templateのトークンIDを取得
    response_token_ids = tokenizer.encode(
        config["response_template"], add_special_tokens=False
    )

    # 各サンプルについて処理
    for i in range(len(labels)):
        input_ids = model_inputs["input_ids"][i]
        response_token_ids_start_idx = None

        # response_templateの開始位置を探す
        for idx in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[idx : idx + len(response_token_ids)] == response_token_ids:
                response_token_ids_start_idx = idx
                break

        # tokenizerのvocabサイズを超えていないか確認
        # for token_id in input_ids:
        #     if token_id > tokenizer.vocab_size:
        #         decode_token = tokenizer.decode(token_id)
        #         raise ValueError(
        #             f"Token ID exceeds tokenizer vocab size: {token_id}, vocab size: {tokenizer.vocab_size}, decode token: {decode_token}"
        #         )

        if response_token_ids_start_idx is None:
            # response_templateが見つからない場合、全てをマスク
            labels[i] = [-100] * len(labels[i])
        else:
            # response_templateの終了位置を計算
            response_token_ids_end_idx = response_token_ids_start_idx + len(
                response_token_ids
            )

            # response_template終了位置までをマスク
            labels[i][:response_token_ids_end_idx] = [-100] * response_token_ids_end_idx

    model_inputs["labels"] = labels

    return model_inputs


def preprocess_function_eval(examples: dict) -> dict:
    """Preprocess the dataset to create input_ids and labels for evaluation.

    Args:
        examples (dict): A dictionary containing the dataset.

    Returns:
        dict: A dictionary containing the preprocessed dataset with input_ids and labels

    """
    # 指示部分
    input_texts = []
    # 全体
    concat_texts = []

    for i in range(len(examples["instruction"])):
        # 指示部分
        instruction_text = (
            f"{examples['instruction'][i]}\n\n{config['response_template']}"
        )
        input_texts.append(instruction_text)

        # 完全な参照テキスト
        concat_text = f"{examples['instruction'][i]}\n\n{config['response_template']}{examples['output'][i]}"  # noqa: E501
        concat_texts.append(concat_text)

    # 入力テキストをトークン化
    model_inputs = tokenizer(
        input_texts,
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )

    # Tokenize the texts
    gold_model_inputs = tokenizer(
        concat_texts,
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )

    # response_template以前を-100でマスクしたlabelsを設定
    labels = gold_model_inputs["input_ids"].copy()

    # response_templateのトークンIDを取得
    response_token_ids = tokenizer.encode(
        config["response_template"], add_special_tokens=False
    )

    # 各サンプルについて処理
    for i in range(len(labels)):
        input_ids = model_inputs["input_ids"][i]
        response_token_ids_start_idx = None

        # response_templateの開始位置を探す
        for idx in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[idx : idx + len(response_token_ids)] == response_token_ids:
                response_token_ids_start_idx = idx
                break

        if response_token_ids_start_idx is None:
            # response_templateが見つからない場合、全てをマスク
            labels[i] = [-100] * len(labels[i])
        else:
            # response_templateの終了位置を計算
            response_token_ids_end_idx = response_token_ids_start_idx + len(
                response_token_ids
            )

            # response_template終了位置までをマスク
            labels[i][:response_token_ids_end_idx] = [-100] * response_token_ids_end_idx

    model_inputs["labels"] = labels

    return model_inputs
