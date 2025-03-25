import json
import re
from pathlib import Path
import numpy as np

import yaml
from transformers import AutoTokenizer, EvalPrediction

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])


def save_dir(preds: list, refs: list, log_dir: str, epoch: int) -> None:
    """Save predictions and references to a file.

    Args:
        preds (list): List of predictions.
        refs (list): List of references.
        log_dir (str): Directory to save the file.
        epoch (int): Epoch number.

    """
    with Path(log_dir + f"/preds_epoch_{epoch}.json").open("w", encoding="utf-8") as f:
        for pred, ref in zip(preds, refs, strict=False):
            instruction = pred.split(config["response_template"])[0].strip()
            pred = pred.split(config["response_template"])[1].strip()
            json.dump(
                {"instruction": instruction, "pred": pred, "ref": ref},
                f,
                ensure_ascii=False,
                indent=4,
            )
            f.write("\n")


def compute_metrics(
    eval_pred: EvalPrediction, log_dir: str, epoch: int
) -> dict[str, float]:
    """Compute metrics for a given set of predictions.

    Args:
        eval_pred (EvalPrediction): A tuple of predictions and labels.
        log_dir (str): Directory to save predictions and references.
        epoch (int): Epoch number.

    Returns:
        dict[str, float]: A dictionary containing the computed metrics.

    """
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Define regex pattern to extract token before ' is a Protein.'
    pattern = re.compile(r"Protein is a (.*?)\.")

    all_recall_denominator = 0
    all_precision_denominator = 0
    all_match_num = 0
    for pred, label in zip(decoded_preds, decoded_labels, strict=False):
        extracted_preds = []
        extracted_labels = []
        pred = pred.split("####")[-1].strip()
        label = label.split("####")[-1].strip()
        for pred_line in pred.splitlines():
            match = pattern.search(pred_line)
            if match:
                extracted_preds.append(match.group(1).strip())
        for label_line in label.splitlines():
            match = pattern.search(label_line)
            if match:
                extracted_labels.append(match.group(1).strip())
        pred_set = set(extracted_preds)
        label_set = set(extracted_labels)
        match_num = len(pred_set & label_set)
        recall_denominator = len(label_set)
        precision_denominator = len(pred_set)
        all_recall_denominator += recall_denominator
        all_precision_denominator += precision_denominator
        all_match_num += match_num

    recall = all_match_num / all_recall_denominator if all_recall_denominator > 0 else 0
    precision = (
        all_match_num / all_precision_denominator
        if all_precision_denominator > 0
        else 0
    )
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0

    save_dir(decoded_preds, decoded_labels, log_dir, epoch)

    return {"eval_recall": recall, "eval_precision": precision, "eval_f1": f1}
