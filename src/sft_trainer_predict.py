import contextlib
from typing import Any

import torch
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel
from trl import SFTTrainer


class SFTTrainerPredict(SFTTrainer):
    """A subclass of SFTTrainer that extends prediction functionality."""
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
        **gen_kwargs: dict[str, Any],
    ):
        """Run prediction and returns predictions and potential metrics.

        This implementation is similar to Seq2SeqTrainer's predict method.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            `PredictionOutput`: The prediction output containing the predictions and metrics.

        """
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and hasattr(self.args, "generation_max_length")
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and hasattr(self.args, "generation_num_beams")
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        return super().predict(
            test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        **gen_kwargs: dict[str, Any],
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
        """Perform an evaluation step on `model` using `inputs`.

        This implementation is similar to Seq2SeqTrainer's prediction_step method.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).

        """
        # Check if predict_with_generate is set and not in prediction_loss_only mode
        if (
            not hasattr(self.args, "predict_with_generate")
            or not self.args.predict_with_generate
            or prediction_loss_only
        ):
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        # Default synced_gpus to True if there's a DeepSpeed or FSDP model
        default_synced_gpus = False
        if (
            hasattr(model, "is_parallelizable")
            and model.is_parallelizable
            and model.model_parallel
        ):
            default_synced_gpus = True
        if hasattr(self, "is_deepspeed_enabled") and self.is_deepspeed_enabled():
            default_synced_gpus = True
        if isinstance(model, FullyShardedDataParallel):
            default_synced_gpus = True

        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        # Prepare generation inputs
        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape
            == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        # For FSDP models, we need to summon full params for generation
        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(model)
            if isinstance(model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            # Generate outputs using model.generate()
            generated_tokens = model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        if hasattr(model, "generation_config") and getattr(
            model.generation_config, "_from_model_config", False
        ):
            model.generation_config._from_model_config = False

        # Retrieve GenerationConfig from model.generation_config
        if hasattr(model, "generation_config"):
            gen_config = model.generation_config
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_config.max_length:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_config.max_length
                )
            elif (
                gen_config.max_new_tokens is not None
                and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
            ):
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_config.max_new_tokens + 1
                )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if hasattr(model, "generation_config"):
                if labels.shape[-1] < gen_config.max_length:
                    labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
                elif (
                    gen_config.max_new_tokens is not None
                    and labels.shape[-1] < gen_config.max_new_tokens + 1
                ):
                    labels = self._pad_tensors_to_max_len(
                        labels, gen_config.max_new_tokens + 1
                    )
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(
        self, tensor: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        """Pad tensors to max_length.

        This implementation is similar to Seq2SeqTrainer's _pad_tensors_to_max_len method.
        """
        if self.processing_class is not None and hasattr(
            self.processing_class, "pad_token_id"
        ):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.processing_class.pad_token_id
                if self.processing_class.pad_token_id is not None
                else self.processing_class.eos_token_id
            )
        elif self.model.config.pad_token_id is not None:
            pad_token_id = self.model.config.pad_token_id
        else:
            raise ValueError("Model configuration missing pad_token_id")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        """Run evaluation and returns metrics.

        This implementation is similar to Seq2SeqTrainer's evaluate method.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.

        """
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and hasattr(self.args, "generation_max_length")
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and hasattr(self.args, "generation_num_beams")
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams

        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # Set predict_with_generate to True for evaluation
        self.args.predict_with_generate = True

        # Call the parent evaluate method
        metrics = super().evaluate(
            eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

        return metrics
