from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer

from utils import formatting_prompts_func

# config
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)


class CustomSFTTrainer(SFTTrainer):
    """標準評価ループをスキップするカスタムSFTTrainer."""

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """評価ループをオーバーライドして、標準の評価を無効化し.

        コールバックのみで評価を実行できるようにする

        Args:
            eval_dataset: 評価データセット
            ignore_keys: 評価時に無視する出力キー
            metric_key_prefix: メトリクス名の接頭辞

        Returns:
            dict[str, float]: 空のメトリクス辞書

        """
        # 空のメトリクス辞書を作成
        metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics


class GenerateEvaluationCallback(TrainerCallback):
    """Custom callback for generating predictions and computing metrics."""

    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        compute_metrics_fn: callable,
        generate_kwargs: dict | None = None,
    ):
        """Initialize the callback.

        Args:
            eval_dataset (Dataset): Evaluation dataset.
            tokenizer (AutoTokenizer): Tokenizer for tokenization.
            compute_metrics_fn (Callable): Function to compute metrics.
            generate_kwargs (dict, optional): Keyword arguments for model generation.

        """
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics_fn = compute_metrics_fn
        self.generate_kwargs = generate_kwargs or {}

    def on_evaluate(
        self,
        args: dict,
        state: dict,
        control: dict,
        model: AutoModelForCausalLM,
        **kwargs,
    ) -> None:
        """Generate predictions and compute metrics.

        Args:
            args: Training arguments.
            state: Training state.
            control: Training control.
            model: Model for prediction.
            **kwargs: Additional keyword arguments.

        """
        # 評価モードに設定
        model.eval()

        # 予測と参照を格納するリスト
        all_preds = []
        all_refs = []

        # バッチごとに処理
        for example in self.eval_dataset:
            all_text = formatting_prompts_func(example)
            # response_templateで区切る
            prompts = [
                text.split(config["response_template"])[0] + config["response_template"]
                for text in all_text
            ]
            references = [
                text.split(config["response_template"])[1] for text in all_text
            ]
            # 入力をトークン化
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # model.generateで生成
            with torch.no_grad():
                outputs = model.generate(**inputs, **self.generate_kwargs)

            # 生成されたテキストをデコード
            decoded_outputs = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            # input_text部分を削除
            decoded_outputs = [
                text.split(config["response_template"])[1] for text in decoded_outputs
            ]
            all_preds.extend(decoded_outputs)
            all_refs.extend(references)

        # 関係トリプルのF1スコアを計算
        metrics = self.compute_metrics_fn((all_preds, all_refs))

        # metricsをTrainerの状態に追加
        for key, value in metrics.items():
            state.log_history[-1][f"eval_{key}"] = value

        return control
