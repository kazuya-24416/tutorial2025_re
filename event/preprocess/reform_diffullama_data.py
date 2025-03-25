import json
from pathlib import Path

TRAIN_DATA = "data/train_genia_ner.json"
EVAL_DATA = "data/dev_genia_ner.json"
TRAIN_DATA_OUT = "data/train_genia_ner.jsonl"
EVAL_DATA_OUT = "data/dev_genia_ner.jsonl"

# instructionの末尾の####を削除する
with Path(TRAIN_DATA).open("r") as f:
    train_data = json.load(f)
    for data in train_data:
        data["instruction"] = data["instruction"].rstrip("####")  # noqa: B005

with Path(EVAL_DATA).open("r") as f:
    eval_data = json.load(f)
    for data in eval_data:
        data["instruction"] = data["instruction"].rstrip("####")  # noqa: B005

# jsonlファイルに保存する
with Path(TRAIN_DATA_OUT).open("w") as f:
    for data in train_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

with Path(EVAL_DATA_OUT).open("w") as f:
    for data in eval_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
