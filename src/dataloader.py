import json
from pathlib import Path

import pandas as pd

INPUT_FILE = "data/debug.jsonl"
OUTPUT_FILE = "data/debug_train.jsonl"


# エンティティにタグを付ける関数を定義
def add_entity_tags(text: str, entities: list) -> str:
    """テキストのエンティティにタグを付ける.

    Args:
        text (str): 元のテキスト
        entities (list): エンティティのリスト

    Returns:
        str: タグ付きテキスト

    """
    # テキストを文字のリストに変換
    chars = list(text)

    # 各位置に挿入する開始タグと終了タグを格納する辞書
    start_tags = {i: [] for i in range(len(chars) + 1)}
    end_tags = {i: [] for i in range(len(chars) + 1)}

    # 各エンティティのスパン情報からタグを準備
    for entity in entities:
        entity_type = entity["type"]

        # 各出現箇所に対してタグを追加
        for i in range(len(entity["start"])):
            start_pos = entity["start"][i]
            end_pos = entity["end"][i]

            # 重複チェック - 既に同じタイプのタグが同じ範囲にあるか確認
            duplicate = False
            for existing_type in start_tags[start_pos]:
                if (
                    existing_type == entity_type
                    and end_pos in end_tags
                    and entity_type in end_tags[end_pos]
                ):
                    duplicate = True
                    break

            # 重複がなければタグを追加
            if not duplicate:
                start_tags[start_pos].append(entity_type)
                end_tags[end_pos].append(entity_type)

    # タグを挿入してテキストを再構築
    tagged_text = ""

    for i in range(len(chars) + 1):
        # 終了タグを追加
        for tag_type in reversed(end_tags[i]):
            tagged_text += f"</{tag_type}>"

        # 開始タグを追加
        for tag_type in start_tags[i]:
            tagged_text += f"<{tag_type}>"

        # 現在位置の文字を追加
        if i < len(chars):
            tagged_text += chars[i]

    return tagged_text


def convert_spans_to_tags_improved(jsonl_file: str, output_file: str) -> None:
    """スパン情報付きのJSONLinesデータを読み込み、テキストにHTMLタグを付けて出力する.

    Args:
        jsonl_file (str): 入力JSONLinesファイルのパス
        output_file (str): 出力ファイルのパス

    """
    tagged_examples = []

    with Path(jsonl_file).open(encoding="utf-8") as f:
        for line in f:
            example = json.loads(line.strip())

            # オリジナルテキスト
            original_text = example["text"]

            # エンティティにタグを付ける
            tagged_text = add_entity_tags(original_text, example["ne_list"])

            # 元のJSONデータに新しくタグ付きテキストを追加
            new_example = {"instruction": tagged_text, "output": []}

            # 関係リストから出力用のトリプルを生成
            for relation in example["relation_list"]:
                subject_id = relation["subject_id"]
                object_id = relation["object_id"]
                relation_type = relation["type"]

                # IDからエンティティテキストを取得
                subject_entity = next(
                    (e["entity"] for e in example["ne_list"] if e["id"] == subject_id),
                    None,
                )
                object_entity = next(
                    (e["entity"] for e in example["ne_list"] if e["id"] == object_id),
                    None,
                )

                if subject_entity and object_entity:
                    triple = f"({subject_entity}, {relation_type}, {object_entity})"
                    new_example["output"].append(triple)

            # 出力を文字列に変換
            new_example["output"] = "\n".join(new_example["output"])

            tagged_examples.append(new_example)

    # 処理したデータを出力
    with Path(output_file).open("w", encoding="utf-8") as f:
        for example in tagged_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def read_jsonlines(file_path: str) -> pd.DataFrame:
    """JSONLinesファイルを読み込み、pandas DataFrameに変換する.

    Args:
        file_path (str): JSONLinesファイルのパス。

    Returns:
        pd.DataFrame: JSONLinesデータを含むDataFrame。

    """
    data = []
    with Path(file_path).open(encoding="utf-8") as f:
        data.extend([json.loads(line.strip()) for line in f])
    return pd.DataFrame(data)


def write_jsonlines(df: pd.DataFrame, file_path: str) -> None:
    """Pandas DataFrameをJSONLinesファイルに書き込む.

    Args:
        df (pd.DataFrame): 書き込むDataFrame。
        file_path (str): 出力ファイルのパス。

    """
    with Path(file_path).open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def convert_to_list_format(input_path: str, output_path: str) -> None:
    """データセットの各行のinstruction と output をリスト形式に変換する.

    Args:
        input_path (str): 入力JSONLinesファイルのパス。
        output_path (str): 出力JSONLinesファイルのパス。

    """
    # データセットを読み込む
    dataset = read_jsonlines(input_path)

    # 各行を変換する
    converted_data = []
    for _, row in dataset.iterrows():
        # 各instructionとoutputを単一要素のリストに変換
        converted_row = {"instruction": [row["instruction"]], "output": [row["output"]]}
        converted_data.append(converted_row)

    # DataFrameに変換して保存
    converted_df = pd.DataFrame(converted_data)
    write_jsonlines(converted_df, output_path)


# 使用例
if __name__ == "__main__":
    # オリジナルのデータセットを作成
    convert_spans_to_tags_improved(INPUT_FILE, OUTPUT_FILE)

    # SFT用にデータセットを変換
    convert_to_list_format(OUTPUT_FILE, OUTPUT_FILE)
