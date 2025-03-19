import warnings
from pathlib import Path

import bitsandbytes as bnb
import pandas as pd
import torch
import transformers
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

model_name_or_path = config["model_name_or_path"]
rank = config["rank"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["bnb_config"]["load_in_4bit"],
    bnb_4bit_use_double_quant=config["bnb_config"]["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=config["bnb_config"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16,
)

warnings.filterwarnings("ignore")

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="cuda:0",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)

model = prepare_model_for_kbit_training(model)


def find_all_linear_names(model: torch.nn.Module, bits: int) -> list[str]:
    """Find all linear names in the model.

    Args:
        model (torch.nn.Module): The model to find linear names in.
        bits (int): The number of bits to use for the linear names.

    Returns:
        list[str]: A list of linear names in the model.

    """
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


target_modules = find_all_linear_names(model, 4)

lora_config = LoraConfig(
    r=rank,
    lora_alpha=rank * 2,
    target_modules=target_modules,
    lora_dropout=config["lora_dropout"],
    bias=config["lora_bias"],
    task_type=config["task_type"],
)

model = get_peft_model(model, lora_config)


data = pd.read_csv(config["data_csv_path"])
data = data.sample(frac=1).reset_index(drop=True)

data = Dataset.from_pandas(data)
data = data.map(
    lambda samples: tokenizer(samples[config["data_column_name"]]), batched=True
)

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(**config["training_arguments"]),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

model.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
