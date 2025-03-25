import json
from pathlib import Path

import pandas as pd
import torch
import yaml

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)


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
        # Format the prompt in a way that's consistent with the model's expected format
        text = f"{example['instruction'][i]}\n\n{config['response_template']}{example['output'][i]}"
        output_texts.append(text)
    return output_texts


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
