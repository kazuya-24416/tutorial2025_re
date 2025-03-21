import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer, EvalPrediction

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)


def save_dir(preds: list, refs: list, log_dir: str) -> None:
    """Save predictions and references to a file.

    Args:
        preds (list): List of predictions.
        refs (list): List of references.
        log_dir (str): Directory to save the file.

    """
    with Path(log_dir + "/preds.json").open("w", encoding="utf-8") as f:
        for pred, ref in zip(preds, refs, strict=False):
            json.dump({"pred": pred, "ref": ref}, f, ensure_ascii=False, indent=4)
            f.write("\n")


# Define custom evaluation metrics for relation triples
def get_compute_metrics_function(tokenizer: AutoTokenizer, log_dir: str) -> Callable:
    """Get compute metrics function for relation triples.

    Args:
        tokenizer (AutoTokenizer): Tokenizer for decoding.
        log_dir (str): Directory to save predictions and references.

    Returns:
        Callable: Compute metrics function.

    """

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        # Get predictions and references
        predictions = eval_pred.predictions
        references = eval_pred.label_ids

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        references = np.where(references != -100, references, tokenizer.pad_token_id)

        # Decode predictions and references
        # tokenizer.pad_token = tokenizer.eos_token
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_refs = tokenizer.batch_decode(
            references, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for pred, ref in zip(decoded_preds, decoded_refs, strict=False):
            print(pred)
            print("---")
            print(ref)
            print("\n")

        save_dir(decoded_preds, decoded_refs, log_dir)

        # Extract triples from predictions and references
        pred_triples_list = []
        ref_triples_list = []

        for pred, _ in zip(decoded_preds, decoded_refs, strict=False):
            # Extract triples from prediction text
            pred_triples = set()
            pred_lines = pred.strip().split("\n")
            for line in pred_lines:
                # Extract triples in format (entity1, relation, entity2)
                triple_match = re.match(r"\((.+?), (.+?), (.+?)\)", line.strip())
                if triple_match:
                    entity1, relation, entity2 = triple_match.groups()
                    pred_triples.add(
                        (entity1.strip(), relation.strip(), entity2.strip())
                    )
            pred_triples_list.append(pred_triples)

        # Extract triples from reference text
        for ref in decoded_refs:
            ref_triples = set()
            ref_lines = ref.strip().split("\n")
            for line in ref_lines:
                triple_match = re.match(r"\((.+?), (.+?), (.+?)\)", line.strip())
                if triple_match:
                    entity1, relation, entity2 = triple_match.groups()
                    ref_triples.add(
                        (entity1.strip(), relation.strip(), entity2.strip())
                    )
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
