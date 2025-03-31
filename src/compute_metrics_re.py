import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer, EvalPrediction

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)


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
            json.dump(
                {"pred": pred, "ref": ref},
                f,
                ensure_ascii=False,
                indent=4,
            )
            f.write("\n")


def compute_metrics(
    eval_pred: EvalPrediction, log_dir: str, epoch: int
) -> dict[str, float]:
    """Compute metrics for relation extraction.

    Args:
        eval_pred (EvalPrediction): Evaluation predictions and references.
        log_dir (str): Directory to save predictions and references.
        epoch (int): Epoch number.

    Returns:
        dict[str, float]: Dictionary of computed metrics.

    """
    # Get predictions and references
    predictions, references = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    references = np.where(references != -100, references, tokenizer.pad_token_id)

    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    decoded_refs = tokenizer.batch_decode(
        references, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # Remove response template from decoded predictions
    decoded_preds = [
        decoded_pred.split(config["response_template"])[1].strip()
        for decoded_pred in decoded_preds
    ]

    save_dir(decoded_preds, decoded_refs, log_dir, epoch)

    # Extract triples from predictions and references
    pred_triples_list = []
    ref_triples_list = []

    for pred, ref in zip(decoded_preds, decoded_refs, strict=False):
        pred_triples = set()
        ref_triples = set()
        pred_lines = pred.strip().split("\n")
        ref_lines = ref.strip().split("\n")
        for pred_line, ref_line in zip(pred_lines, ref_lines, strict=False):
            # Extract triples in format (entity1, relation, entity2)
            pred_triple_match = re.match(r"\((.+?), (.+?), (.+?)\)", pred_line.strip())
            if pred_triple_match:
                entity1, relation, entity2 = pred_triple_match.groups()
                pred_triples.add((entity1.strip(), relation.strip(), entity2.strip()))
            ref_triple_match = re.match(r"\((.+?), (.+?), (.+?)\)", ref_line.strip())
            if ref_triple_match:
                entity1, relation, entity2 = ref_triple_match.groups()
                ref_triples.add((entity1.strip(), relation.strip(), entity2.strip()))

        pred_triples_list.append(pred_triples)
        ref_triples_list.append(ref_triples)

    # Calculate precision, recall, and F1 score
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for pred_triples, ref_triples in zip(
        pred_triples_list, ref_triples_list, strict=False
    ):
        # Calculate true positives (intersection of pred and ref)
        true_positives = len(pred_triples.intersection(ref_triples))

        # Calculate precision, recall, and F1
        precision = true_positives / max(len(pred_triples), 1)
        recall = true_positives / max(len(ref_triples), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate average metrics
    avg_precision = total_precision / max(len(pred_triples_list), 1)
    avg_recall = total_recall / max(len(ref_triples_list), 1)
    avg_f1 = total_f1 / max(len(pred_triples_list), 1)

    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}


def get_compute_metrics(tokenizer: AutoTokenizer, log_dir: str) -> Callable:
    """Get compute metrics function."""

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        # Get predictions and references
        predictions, references = eval_pred

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        references = np.where(references != -100, references, tokenizer.pad_token_id)

        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_refs = tokenizer.batch_decode(
            references, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # Remove response template from decoded predictions
        instructions = [
            decoded_ref.split(config["response_template"])[0].strip()
            for decoded_ref in decoded_refs
        ]
        decoded_refs = [
            decoded_ref.split(config["response_template"])[1].strip()
            for decoded_ref in decoded_refs
        ]
        decoded_preds = [
            decoded_pred.split(config["response_template"])[1].strip()
            for decoded_pred in decoded_preds
        ]

        # ログを保存
        with Path(log_dir + f"/preds_{time.time()}.json").open(
            "w", encoding="utf-8"
        ) as f:
            for instruction, pred, ref in zip(
                instructions, decoded_preds, decoded_refs, strict=False
            ):
                json.dump(
                    {"instruction": instruction, "pred": pred, "ref": ref},
                    f,
                    ensure_ascii=False,
                    indent=3,
                )
                f.write("\n")

        # Extract triples from predictions and references
        pred_triples_list = []
        ref_triples_list = []

        for pred, ref in zip(decoded_preds, decoded_refs, strict=False):
            pred_triples = set()
            ref_triples = set()
            pred_lines = pred.strip().split("\n")
            ref_lines = ref.strip().split("\n")
            for pred_line, ref_line in zip(pred_lines, ref_lines, strict=False):
                # Extract triples in format (entity1, relation, entity2)
                pred_triple_match = re.match(
                    r"\((.+?), (.+?), (.+?)\)", pred_line.strip()
                )
                if pred_triple_match:
                    entity1, relation, entity2 = pred_triple_match.groups()
                    pred_triples.add(
                        (entity1.strip(), relation.strip(), entity2.strip())
                    )
                ref_triple_match = re.match(
                    r"\((.+?), (.+?), (.+?)\)", ref_line.strip()
                )
                if ref_triple_match:
                    entity1, relation, entity2 = ref_triple_match.groups()
                    ref_triples.add(
                        (entity1.strip(), relation.strip(), entity2.strip())
                    )

            pred_triples_list.append(pred_triples)
            ref_triples_list.append(ref_triples)

        # Calculate precision, recall, and F1 score
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for pred_triples, ref_triples in zip(
            pred_triples_list, ref_triples_list, strict=False
        ):
            # Calculate true positives (intersection of pred and ref)
            true_positives = len(pred_triples.intersection(ref_triples))

            # Calculate precision, recall, and F1
            precision = true_positives / max(len(pred_triples), 1)
            recall = true_positives / max(len(ref_triples), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        # Calculate average metrics
        avg_precision = total_precision / max(len(pred_triples_list), 1)
        avg_recall = total_recall / max(len(ref_triples_list), 1)
        avg_f1 = total_f1 / max(len(pred_triples_list), 1)

        return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}

    return compute_metrics
