import copy
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from compute_metrics_re import get_compute_metrics_function
from utils import formatting_prompts_func, preprocess_logits_for_metrics, read_jsonlines

# Load configuration from config.yaml
with Path("config/config.yaml").open() as file:
    config = yaml.safe_load(file)

# Extract configuration values
model_name_or_path = config["model_name_or_path"]
dataset_path = config["dataset_path"]
response_template = config["response_template"]
use_lora = config.get("use_lora", False)
use_quantization = config.get("use_quantization", False)

dataset = read_jsonlines(dataset_path)
dataset = Dataset.from_pandas(dataset)

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Setup quantization configuration if enabled
quantization_kwargs = {}
if use_quantization:
    quant_config = config["quantization_config"]
    bits = quant_config.get("bits", 8)  # Default to 8-bit if not specified

    # Configure quantization based on bits
    if bits == 4:
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=getattr(
                torch, quant_config.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_use_double_quant=quant_config.get(
                "bnb_4bit_use_double_quant", True
            ),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        )
    else:  # 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            llm_int8_threshold=quant_config.get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=quant_config.get(
                "llm_int8_has_fp16_weight", False
            ),
        )

    quantization_kwargs["quantization_config"] = bnb_config

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,  # Use float16 for better performance
    device_map="auto",  # Automatically distribute model across available GPUs
    **quantization_kwargs,
)

# Apply LoRA if enabled
if use_lora:
    # Prepare model for k-bit training if quantization is enabled
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora_config"].get("r", 16),
        lora_alpha=config["lora_config"].get("lora_alpha", 32),
        lora_dropout=config["lora_config"].get("lora_dropout", 0.05),
        bias=config["lora_config"].get("bias", "none"),
        task_type=config["lora_config"].get("task_type", "CAUSAL_LM"),
        target_modules=config["lora_config"].get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters info

# Create data collator with response template from config
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# eval
hprams_name = f"""
    lr{config["training_args"]["learning_rate"]}_bs{config["training_args"]["per_device_train_batch_size"]}
    """
log_dir = config["training_args"]["output_dir"] + f"/{hprams_name}"
Path(log_dir).mkdir(parents=True, exist_ok=True)
compute_metrics = get_compute_metrics_function(tokenizer, log_dir)

# Get training arguments from config
training_args = config["training_args"]

training_args_copy = copy.deepcopy(training_args)

# Configure Weights & Biases
wandb_config = config.get("wandb", {})
wandb_enabled = wandb_config.get("enabled", True)

# Update report_to based on wandb configuration
if not wandb_enabled:
    # Disable wandb during debugging
    if "report_to" in training_args and "wandb" in training_args["report_to"]:
        if training_args["report_to"] == "wandb":
            training_args["report_to"] = "none"
        elif training_args["report_to"] == "all":
            training_args["report_to"] = ["tensorboard"]
        elif (
            isinstance(training_args["report_to"], list)
            and "wandb" in training_args["report_to"]
        ):
            training_args["report_to"] = [
                r for r in training_args["report_to"] if r != "wandb"
            ]
else:
    # Initialize wandb with project settings
    wandb.init(
        project=wandb_config.get("project", "tutorial2025_re"),
        name=wandb_config.get("name", None),
        group=wandb_config.get("group", None),
        tags=wandb_config.get("tags", []),
        mode="disabled" if wandb_config.get("debug", False) else "online",
    )


# カスタム評価のためのSFTTrainerのサブクラス
class CustomSFTTrainer(SFTTrainer):
    """トレーニング中の評価ステップをオーバーライドして.

    レスポンス部分だけを生成・評価するようにします.
    """

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """トレーニング中の評価ステップをオーバーライドして.

        レスポンス部分だけを生成・評価するようにします.

        Args:
            model (torch.nn.Module): 評価するモデル
            inputs (dict): 入力データ
            prediction_loss_only (bool): 予測損失のみを計算するかどうか
            ignore_keys (Optional[List[str]], optional): 忽略するキーワード. Defaults to None.

        Returns:
            tuple: (loss, logits, labels, predictions)

        """
        # predict_with_generateが有効で予測のみを行う場合
        if self.args.predict_with_generate and not prediction_loss_only:
            # 入力と出力（ラベル）を準備
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)

            # 生成用の入力を取得（レスポンステンプレートまで）
            prompt_input_ids = inputs["input_ids"].clone()

            # レスポンステンプレートの位置を特定
            template_ids = self.tokenizer.encode(
                response_template, add_special_tokens=False
            )
            response_token_ids = []

            # 各バッチ要素ごとにテンプレート以降の位置を特定
            max_response_length = 0
            for i, ids in enumerate(prompt_input_ids):
                ids_list = ids.tolist()

                # テンプレートの位置を検索
                for j in range(len(ids_list) - len(template_ids) + 1):
                    if ids_list[j : j + len(template_ids)] == template_ids:
                        template_pos = j + len(template_ids)
                        prompt_input_ids[i, template_pos:] = self.tokenizer.pad_token_id
                        response_length = len(ids_list) - template_pos
                        max_response_length = max(max_response_length, response_length)
                        response_token_ids.append((i, template_pos))
                        break

            # 文章生成の設定
            gen_kwargs = {
                "max_length": prompt_input_ids.shape[1]
                + max_response_length
                + 50,  # 少し余裕を持たせる
                "num_beams": self.args.generation_num_beams
                if hasattr(self.args, "generation_num_beams")
                else 1,
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.7,
            }

            # レスポンステンプレート以降を生成
            generated_tokens = self.model.generate(
                prompt_input_ids,
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

            # 生成されたトークンから、テンプレート以降の部分だけを抽出
            prediction_ids = []
            for i, template_pos in response_token_ids:
                # 元の入力長以降の部分（生成部分）を取得
                pred = generated_tokens[i, template_pos:].unsqueeze(0)
                prediction_ids.append(pred)

            if len(prediction_ids) > 0:
                prediction_ids = torch.cat(prediction_ids, dim=0)

            # ラベル（正解）の取得
            if has_labels:
                labels = tuple(inputs.get(name) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            return (None, prediction_ids, labels)

        # 通常のprediction_stepに委任（損失計算時など）
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def _prepare_inputs(self, inputs):
        """入力を処理するメソッド"""
        # 親クラスの処理を呼び出す
        inputs = super()._prepare_inputs(inputs)
        return inputs


# Create SFTConfig from training arguments
args = SFTConfig(**training_args_copy)

# Create custom SFT trainer
trainer = CustomSFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,  # 同じデータセットを評価用にも使用
    args=args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=lora_config if use_lora else None,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    tokenizer=tokenizer,  # トークナイザーを明示的に渡す
)

# 評価時にgenerateを使うように直接設定
trainer.args.predict_with_generate = True
trainer.args.generation_max_length = 1024
trainer.args.generation_num_beams = 1

# Start training
trainer.train()
