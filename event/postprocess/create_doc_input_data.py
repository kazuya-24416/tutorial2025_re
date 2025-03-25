#!/usr/bin/env python
# create_doc_input_data.py

import json
import os
from collections import defaultdict
from typing import Any


def create_input_data_by_doc_id(data_path: str) -> dict[str, list[dict[str, Any]]]:
    """LLaMA-Factory/data/dev_genia_ee.jsonのデータからdoc_idごとにinput_dataを作成する関数

    Args:
        data_path: 入力データのパス（例: "LLaMA-Factory/data/dev_genia_ee.json"）

    Returns:
        doc_idをキー、そのdoc_idに属するデータのリストを値とする辞書

    """
    # データを読み込む
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    # doc_idごとにデータをグループ化
    doc_data = defaultdict(list)
    for item in data:
        doc_id = item.get("doc_id")
        if doc_id:
            doc_data[doc_id].append(item)

    return dict(doc_data)


def save_input_data_by_doc_id(
    data_path: str, output_dir: str = "LLaMA-Factory/data/doc_input_data"
):
    """doc_idごとにグループ化したデータを個別のJSONファイルとして保存する関数

    Args:
        data_path: 入力データのパス（例: "LLaMA-Factory/data/dev_genia_ee.json"）
        output_dir: 出力ディレクトリのパス

    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # doc_idごとにデータをグループ化
    doc_data = create_input_data_by_doc_id(data_path)

    # 各doc_idごとにファイルを保存
    for doc_id, items in doc_data.items():
        # ファイル名に使用できない文字を置換
        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_")
        output_path = os.path.join(output_dir, f"{safe_doc_id}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"保存完了: {len(doc_data)}件のdoc_idデータを{output_dir}に保存しました")
    return doc_data


def get_input_data_for_doc_id(
    data_path: str, target_doc_id: str
) -> list[dict[str, Any]]:
    """特定のdoc_idに対応するinput_dataを取得する関数

    Args:
        data_path: 入力データのパス（例: "LLaMA-Factory/data/dev_genia_ee.json"）
        target_doc_id: 取得したいdoc_id

    Returns:
        指定されたdoc_idに対応するデータのリスト

    """
    # doc_idごとにデータをグループ化
    doc_data = create_input_data_by_doc_id(data_path)

    # 指定されたdoc_idのデータを返す
    return doc_data.get(target_doc_id, [])


def list_available_doc_ids(data_path: str) -> list[str]:
    """データファイル内の利用可能なdoc_idのリストを取得する関数

    Args:
        data_path: 入力データのパス（例: "LLaMA-Factory/data/dev_genia_ee.json"）

    Returns:
        利用可能なdoc_idのリスト

    """
    # doc_idごとにデータをグループ化
    doc_data = create_input_data_by_doc_id(data_path)

    # doc_idのリストを返す
    return list(doc_data.keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="doc_idごとにinput_dataを作成するスクリプト"
    )
    parser.add_argument(
        "--data_path",
        default="LLaMA-Factory/data/dev_genia_ee.json",
        help="入力データのパス",
    )
    parser.add_argument(
        "--output_dir",
        default="LLaMA-Factory/data/doc_input_data",
        help="出力ディレクトリのパス",
    )
    parser.add_argument(
        "--list_doc_ids", action="store_true", help="利用可能なdoc_idのリストを表示"
    )
    parser.add_argument("--doc_id", help="特定のdoc_idのデータを表示")

    args = parser.parse_args()

    if args.list_doc_ids:
        doc_ids = list_available_doc_ids(args.data_path)
        print(f"利用可能なdoc_id ({len(doc_ids)}件):")
        for doc_id in sorted(doc_ids):
            print(f"  - {doc_id}")

    elif args.doc_id:
        items = get_input_data_for_doc_id(args.data_path, args.doc_id)
        print(f"{args.doc_id}のデータ ({len(items)}件):")
        print(json.dumps(items, ensure_ascii=False, indent=2))

    else:
        # デフォルトの動作: すべてのdoc_idごとにファイルを保存
        doc_data = save_input_data_by_doc_id(args.data_path, args.output_dir)
        print(f"doc_idの件数: {len(doc_data)}")
        print(f"データ項目の合計件数: {sum(len(items) for items in doc_data.values())}")
