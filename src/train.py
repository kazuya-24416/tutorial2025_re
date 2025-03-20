import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Extract configuration values
model_name_or_path = config["model_name_or_path"]
dataset_path = config["dataset_path"]
response_template = config["response_template"]


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


dataset = read_jsonlines(dataset_path)
dataset = Dataset.from_pandas(dataset)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def formatting_prompts_func(example: dict) -> list:
    """Format prompts for training.

    Args:
        example (dict): 1つの例のデータ.

    Returns:
        list: フォーマット済みのテキストのリスト.

    """
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"""
        ### Instruction: {example["instruction"][i]}\n
        ### Answer: {example["output"][i]}"""
        output_texts.append(text)
    return output_texts


# Create data collator with response template from config
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Create SFTConfig from training arguments in config
training_args = config["training_args"]
args = SFTConfig(**training_args)


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    eval_dataset=dataset,
    args=args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
