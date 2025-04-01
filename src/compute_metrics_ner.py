import json
import os
import re
import time

from typing import Callable
from transformers import EvalPrediction, PreTrainedTokenizer


def get_compute_metrics_function(
    tokenizer: PreTrainedTokenizer, output_dir: str
) -> Callable:
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = eval_pred
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

        recall = (
            all_match_num / all_recall_denominator if all_recall_denominator > 0 else 0
        )
        precision = (
            all_match_num / all_precision_denominator
            if all_precision_denominator > 0
            else 0
        )
        f1 = (
            2 * recall * precision / (recall + precision)
            if recall + precision > 0
            else 0
        )

        # evalのログを保存
        dt_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        output_prediction_file = os.path.join(
            output_dir, f"generated_predictions_{dt_now}.jsonl"
        )
        with open(output_prediction_file, "a", encoding="utf-8") as writer:
            res: list[str] = []
            for pred, label in zip(decoded_preds, decoded_labels, strict=False):
                res.append(
                    json.dumps({"predict": pred, "label": label}, ensure_ascii=False)
                )
            writer.write("\n".join(res))

        return {"eval_recall": recall, "eval_precision": precision, "eval_f1": f1}

    return compute_metrics
