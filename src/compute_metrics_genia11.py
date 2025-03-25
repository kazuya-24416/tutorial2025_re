import json
import os
import time
from pathlib import Path

from event.postprocess.run_converter import convert_and_save_data, run_evaluation
from transformers import EvalPrediction, PreTrainedTokenizer


def get_compute_metrics_function(
    tokenizer: PreTrainedTokenizer, output_dir: str, metadata_dict: dict[str, str]
) -> callable:
    """Return a function that computes metrics for a given set of predictions.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
        output_dir (str): The directory where the predictions and evaluation results will be saved.
        metadata_dict (dict[str, str]): A dictionary containing metadata for each input text.

    Returns:
        callable: A function that computes metrics for a given set of predictions.

    """
    # Create output directory for predictions
    predictions_dir = Path(output_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # 評価用のディレクトリを作成
    eval_output_dir = Path(output_dir) / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # 参照データのディレクトリ（goldデータのディレクトリパス）
    gold_data_dir = "event/ge11/BioNLP-ST_2011_genia_devel_data_rev1"

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """Compute metrics for a given set of predictions.

        Args:
            eval_pred (EvalPrediction): A tuple of predictions and labels.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics.

        """
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # 予測結果をログに保存
        current_time = time.localtime()
        timestamp = time.strftime("%m-%d_%H:%M", current_time)
        log_file = os.path.join(predictions_dir, f"log_{timestamp}.json")

        # 予測結果をJSONファイルに保存
        with open(log_file, "w", encoding="utf-8") as writer:
            log_data = []
            for pred, label in zip(decoded_preds, decoded_labels, strict=False):
                input_text = label.split("####")[0]
                labels_text = (
                    label.split("####")[1] if len(label.split("####")) > 1 else ""
                )
                pred_text = pred.split("####")[1] if len(pred.split("####")) > 1 else ""

                metadata = metadata_dict.get(input_text, {})
                doc_id = metadata.get("doc_id")
                wnd_id = metadata.get("wnd_id")

                prediction_item = {
                    "doc_id": doc_id,
                    "wnd_id": wnd_id,
                    "instruction": input_text,
                    "labels": labels_text,
                    "output": pred_text,
                }
                log_data.append(prediction_item)
            json.dump(log_data, writer, ensure_ascii=False, indent=4)

        convert_and_save_data(
            "event/ge11_json/dev.json",
            log_file,
            "event/output/converted_all.json",
            "event/output/reverted_files",
        )

        # run_evaluationを使用して評価を実行
        scores = run_evaluation(
            ref_dir=gold_data_dir, pred_dir="event/output/reverted_files", verbose=True
        )
        precision = scores["TOTAL"]["precision"]
        recall = scores["TOTAL"]["recall"]
        f1 = scores["TOTAL"]["fscore"]

        return {"eval_precision": precision, "eval_recall": recall, "eval_f1": f1}

    return compute_metrics
