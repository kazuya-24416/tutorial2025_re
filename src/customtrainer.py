# import torch
# from datasets import Dataset
# from transformers import (
#     PreTrainedModel,
#     Seq2SeqTrainer,
#     Seq2SeqTrainingArguments,
# )
# from transformers.data import DataCollatorForSeq2Seq
# from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


# class CombinedTrainer(SFTTrainer, Seq2SeqTrainer):
#     """A trainer class that combines SFTTrainer and Seq2SeqTrainer functionality.

#     It allows using DataCollatorForCompletionOnlyLM and predict_with_generate
#     """

#     def __init__(
#         self,
#         model: PreTrainedModel | None = None,
#         args: Seq2SeqTrainingArguments | None = None,
#         data_collator: DataCollatorForSeq2Seq | None = None,
#         train_dataset: Dataset | None = None,
#         eval_dataset: Dataset | None = None,
#         tokenizer=None,
#         model_init=None,
#         compute_metrics=None,
#         callbacks=None,
#         optimizers=(None, None),
#         preprocess_logits_for_metrics=None,
#         peft_config=None,
#         dataset_text_field=None,
#         max_seq_length=None,
#         packing=False,
#         formatting_func=None,
#         response_template=None,
#     ):
#         # Initialize SFTTrainer
#         SFTTrainer.__init__(
#             self,
#             model=model,
#             args=args,
#             data_collator=data_collator,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             processing_class=tokenizer,
#             # model_init=model_init,
#             compute_metrics=compute_metrics,
#             callbacks=callbacks,
#             optimizers=optimizers,
#             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#             peft_config=peft_config,
#             # dataset_text_field=dataset_text_field,
#             # max_seq_length=max_seq_length,
#             # packing=packing,
#             formatting_func=formatting_func,
#         )

#         # Store response template for DataCollatorForCompletionOnlyLM
#         self.response_template = response_template

#         # Set up DataCollatorForCompletionOnlyLM if needed
#         if (
#             isinstance(data_collator, DataCollatorForCompletionOnlyLM)
#             or data_collator is None
#         ):
#             if response_template is not None and tokenizer is not None:
#                 self.data_collator = DataCollatorForCompletionOnlyLM(
#                     tokenizer=tokenizer,
#                     response_template=response_template,
#                     mlm=False,
#                     pad_to_multiple_of=8 if self.use_cuda_amp else None,
#                 )

#     def prediction_step(
#         self,
#         model,
#         inputs,
#         prediction_loss_only,
#         ignore_keys=None,
#     ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
#         """Override the prediction_step to use Seq2SeqTrainer's functionality when needed"""
#         if self.args.predict_with_generate and not prediction_loss_only:
#             return Seq2SeqTrainer.prediction_step(
#                 self, model, inputs, prediction_loss_only, ignore_keys=ignore_keys
#             )
#         return SFTTrainer.prediction_step(
#             self, model, inputs, prediction_loss_only, ignore_keys=ignore_keys
#         )

#     def evaluate(
#         self,
#         eval_dataset=None,
#         ignore_keys=None,
#         metric_key_prefix="eval",
#     ) -> dict[str, float]:
#         """Override evaluate to support generation-based evaluation"""
#         if self.args.predict_with_generate:
#             return Seq2SeqTrainer.evaluate(
#                 self,
#                 eval_dataset=eval_dataset,
#                 ignore_keys=ignore_keys,
#                 metric_key_prefix=metric_key_prefix,
#             )
#         return SFTTrainer.evaluate(
#             self,
#             eval_dataset=eval_dataset,
#             ignore_keys=ignore_keys,
#             metric_key_prefix=metric_key_prefix,
#         )
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.data import DataCollatorForSeq2Seq
from trl import DataCollatorForCompletionOnlyLM


class CombinedTrainer(Seq2SeqTrainer):
    """A trainer class that combines Seq2SeqTrainer functionality with SFT features.
    It allows using DataCollatorForCompletionOnlyLM and predict_with_generate
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        args: Seq2SeqTrainingArguments | None = None,
        data_collator: DataCollatorForSeq2Seq | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        dataset_text_field=None,
        max_seq_length=None,
        packing=False,
        formatting_func=None,
        response_template=None,
    ):
        # Initialize Seq2SeqTrainer with processing_class instead of tokenizer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,  # この警告は無視してOK。内部でtokenizerとprocessing_classを整理する
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Store SFT-specific attributes
        self.peft_config = peft_config
        self.dataset_text_field = dataset_text_field
        self.max_seq_length = max_seq_length
        self.packing = packing
        self.formatting_func = formatting_func
        self.response_template = response_template

        # Set up DataCollatorForCompletionOnlyLM if needed
        if (
            isinstance(data_collator, DataCollatorForCompletionOnlyLM)
            or data_collator is None
        ):
            if response_template is not None and tokenizer is not None:
                # fp16/bf16設定からpad_to_multipleを判断
                pad_to_multiple_of = None
                if (hasattr(self.args, "fp16") and self.args.fp16) or (
                    hasattr(self.args, "bf16") and self.args.bf16
                ):
                    pad_to_multiple_of = 8

                self.data_collator = DataCollatorForCompletionOnlyLM(
                    tokenizer=tokenizer,
                    response_template=response_template,
                    mlm=False,
                    # pad_to_multiple_of=pad_to_multiple_of,
                )

    def train(self, *args, **kwargs):
        """Run training with optional PEFT setup"""
        # Support PEFT integration if needed
        if hasattr(self, "peft_config") and self.peft_config is not None:
            try:
                from peft import get_peft_model, prepare_model_for_kbit_training

                # Check if model is quantized
                is_qlora = getattr(self.model, "is_loaded_in_4bit", False) or getattr(
                    self.model, "is_loaded_in_8bit", False
                )

                if is_qlora:
                    self.model = prepare_model_for_kbit_training(
                        self.model,
                        use_gradient_checkpointing=self.args.gradient_checkpointing
                        if hasattr(self.args, "gradient_checkpointing")
                        else False,
                    )

                # Apply PEFT
                self.model = get_peft_model(self.model, self.peft_config)
            except ImportError:
                print("PEFT library not found. Skipping PEFT integration.")

        return super().train(*args, **kwargs)
