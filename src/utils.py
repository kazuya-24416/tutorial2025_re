import json
from pathlib import Path

import pandas as pd


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


def formatting_prompts_func(example: dict) -> list:
    """Format prompts for training.

    Args:
        example (dict): 1つの例のデータ.

    Returns:
        list: フォーマット済みのテキストのリスト.

    """
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"""
        ### Instruction: {example["instruction"][i]}\n
        ### Answer: {example["output"][i]}"""
        output_texts.append(text)
    return output_texts
