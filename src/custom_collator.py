import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define a custom data collator that masks the instruction part with -100
class CustomDataCollatorForSeq2Seq:
    """Custom data collator for sequence-to-sequence tasks."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        response_template: str,
        padding: str = "max_length",
        max_length: int = 512,
    ) -> None:
        """Initialize the custom data collator.

        Args:
            tokenizer: The tokenizer to use for tokenization.
            model: The model to use for tokenization.
            response_template: The response template to use for tokenization.
            padding: The padding to use for tokenization.
            max_length: The maximum length to use for tokenization.

        """
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.response_template = response_template
        # Tokenize the response template to find its start in the sequence
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, features: list[dict[str, int]]) -> dict[str, torch.Tensor]:
        """Tokenize and pad the input features.

        Args:
            features: List of features to be tokenized and padded.

        Returns:
            A dictionary of tokenized and padded features.

        """
        # Tokenizer's default collation
        batch = {}

        # Handling input_ids and attention_mask
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]

        # Pad input_ids and attention_mask
        if self.padding == "max_length":
            input_ids = [
                ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
                if len(ids) < self.max_length
                else ids[: self.max_length]
                for ids in input_ids
            ]
            attention_mask = [
                mask + [0] * (self.max_length - len(mask))
                if len(mask) < self.max_length
                else mask[: self.max_length]
                for mask in attention_mask
            ]

        batch["input_ids"] = torch.tensor(input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)

        # Create labels with -100 for instruction part
        labels = []
        for feature_input_ids in input_ids:
            label = feature_input_ids.copy()

            # Find the position of response_template in the sequence
            response_start_idx = -1
            for i in range(len(label) - len(self.response_token_ids) + 1):
                if (
                    label[i : i + len(self.response_token_ids)]
                    == self.response_token_ids
                ):
                    response_start_idx = i
                    break

            # If response template is found, mask everything before it with -100
            if response_start_idx != -1:
                label[:response_start_idx] = [-100] * response_start_idx

            labels.append(label)

        batch["labels"] = torch.tensor(labels)

        return batch
