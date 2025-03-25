#!/usr/bin/env python
# extract_mentions.py

import json
import os
from collections import defaultdict
from typing import Any


def extract_sentence_start_from_json(data_path: str) -> dict[str, list[int]]:
    """event/ge11_json/dev.jsonから文の開始位置を抽出する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）

    Returns:
        doc_idをキー、そのdoc_idに属するwindowごとの文の開始位置のリストを値とする辞書

    """
    # データを読み込む
    with open(data_path, encoding="utf-8") as f:
        # Check first character to determine if it's a JSON array or JSON Lines
        first_char = f.read(1)
        f.seek(0)  # Reset file pointer
        if first_char == "[":
            # JSON array
            data = json.load(f)
        else:
            # JSON Lines
            data = [json.loads(line) for line in f]

    # doc_idごとにwindowごとの文の開始位置を格納する辞書を初期化
    sentence_start_dict = defaultdict(list)

    # 各windowの開始位置を抽出して辞書に格納
    for item in data:
        doc_id = item.get("doc_id")
        if doc_id:
            start = item.get("sentence_start")
            if start is not None:
                sentence_start_dict[doc_id].append(start)

    return dict(sentence_start_dict)


def extract_sentence_from_json(data_path: str) -> dict[str, list[str]]:
    """event/ge11_json/dev.jsonから文を抽出する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）

    Returns:
        doc_idをキー、そのdoc_idに属するwindowごとの文のリストを値とする辞書

    """
    # データを読み込む
    with open(data_path, encoding="utf-8") as f:
        # Check first character to determine if it's a JSON array or JSON Lines
        first_char = f.read(1)
        f.seek(0)  # Reset file pointer

        if first_char == "[":
            # Standard JSON array format
            data = json.load(f)
        else:
            # JSON Lines format (one JSON object per line)
            data = []
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))

    # doc_idごとにデータをグループ化
    doc_sentences = defaultdict(list)

    # 現在のdoc_idとそのwindowのリスト
    current_doc_id = None
    current_sentences = []
    # 各文を処理
    for item in data:
        if "doc_id" in item:
            if current_doc_id != item["doc_id"]:
                if current_doc_id is not None:
                    doc_sentences[current_doc_id] = current_sentences
                current_doc_id = item["doc_id"]
                current_sentences = []
        if "sentence" in item:
            current_sentences.append(item["sentence"])

    # 最後のdoc_idに対する文のリストを追加
    if current_doc_id is not None:
        doc_sentences[current_doc_id] = current_sentences

    return dict(doc_sentences)


def extract_mentions_from_json(
    data_path: str,
) -> tuple[
    dict[str, list[list[dict[str, Any]]]], dict[str, list[list[dict[str, Any]]]]
]:
    """event/ge11_json/dev.jsonからentity_mentionsとevent_mentionsを抽出する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）

    Returns:
        Tuple containing:
        - doc_idをキー、そのdoc_idに属するwindowごとのentity_mentionsのリストを値とする辞書
        - doc_idをキー、そのdoc_idに属するwindowごとのevent_mentionsのリストを値とする辞書

    """
    # データを読み込む
    with open(data_path, encoding="utf-8") as f:
        # Check first character to determine if it's a JSON array or JSON Lines
        first_char = f.read(1)
        f.seek(0)  # Reset file pointer

        if first_char == "[":
            # Standard JSON array format
            data = json.load(f)
        else:
            # JSON Lines format (one JSON object per line)
            data = []
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))

    # doc_idごとにデータをグループ化
    doc_entity_mentions = defaultdict(list)
    doc_event_mentions = defaultdict(list)

    # 現在のdoc_idとそのwindowのリスト
    current_doc_id = None
    current_entity_mentions = []
    current_event_mentions = []

    for item in data:
        doc_id = item.get("doc_id")
        entity_mentions = item.get("entity_mentions", [])
        event_mentions = item.get("event_mentions", [])

        # 新しいdoc_idに切り替わった場合、前のdoc_idのデータを保存
        if current_doc_id is not None and current_doc_id != doc_id:
            if current_entity_mentions:
                doc_entity_mentions[current_doc_id] = current_entity_mentions
            if current_event_mentions:
                doc_event_mentions[current_doc_id] = current_event_mentions
            current_entity_mentions = []
            current_event_mentions = []

        current_doc_id = doc_id
        current_entity_mentions.append(entity_mentions)
        current_event_mentions.append(event_mentions)

    # 最後のdoc_idのデータを保存
    if current_doc_id is not None:
        if current_entity_mentions:
            doc_entity_mentions[current_doc_id] = current_entity_mentions
        if current_event_mentions:
            doc_event_mentions[current_doc_id] = current_event_mentions

    return dict(doc_entity_mentions), dict(doc_event_mentions)


def save_mentions_by_doc_id(data_path: str, output_dir: str = "event/output/mentions"):
    """doc_idごとにグループ化したentity_mentionsとevent_mentionsを個別のJSONファイルとして保存する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）
        output_dir: 出力ディレクトリのパス

    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # entity_mentionsとevent_mentionsを抽出
    doc_entity_mentions, doc_event_mentions = extract_mentions_from_json(data_path)

    # 各doc_idごとにファイルを保存
    for doc_id in doc_entity_mentions.keys():
        # ファイル名に使用できない文字を置換
        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_")

        # entity_mentionsを保存
        entity_output_path = os.path.join(
            output_dir, f"{safe_doc_id}_entity_mentions.json"
        )
        with open(entity_output_path, "w", encoding="utf-8") as f:
            json.dump(doc_entity_mentions[doc_id], f, ensure_ascii=False, indent=2)

        # event_mentionsを保存
        event_output_path = os.path.join(
            output_dir, f"{safe_doc_id}_event_mentions.json"
        )
        with open(event_output_path, "w", encoding="utf-8") as f:
            json.dump(doc_event_mentions[doc_id], f, ensure_ascii=False, indent=2)

    print(
        f"保存完了: {len(doc_entity_mentions)}件のdoc_idのmentionsデータを{output_dir}に保存しました"
    )
    return doc_entity_mentions, doc_event_mentions


def get_sentence_start_for_doc_id(data_path: str, target_doc_id: str) -> list[int]:
    """特定のdoc_idに対応するwindowごとの文の開始位置を取得する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）
        target_doc_id: 取得したいdoc_id

    Returns:
        doc_idに対応するwindowごとの文の開始位置のリスト

    """
    return extract_sentence_start_from_json(data_path).get(target_doc_id, [])


def get_sentence_for_doc_id(data_path: str, target_doc_id: str) -> dict[str, list[str]]:
    """特定のdoc_idに対応する文を取得する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）
        target_doc_id: 取得したいdoc_id

    Returns:
        doc_idに対応するwindowごとの文のリスト

    """
    return extract_sentence_from_json(data_path).get(target_doc_id, [])


def get_mentions_for_doc_id(
    data_path: str, target_doc_id: str
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """特定のdoc_idに対応するentity_mentionsとevent_mentionsを取得する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）
        target_doc_id: 取得したいdoc_id

    Returns:
        Tuple containing:
        - 指定されたdoc_idに対応するwindowごとのentity_mentionsのリスト
        - 指定されたdoc_idに対応するwindowごとのevent_mentionsのリスト

    """
    # entity_mentionsとevent_mentionsを抽出
    doc_entity_mentions, doc_event_mentions = extract_mentions_from_json(data_path)

    # 指定されたdoc_idのデータを返す
    return doc_entity_mentions.get(target_doc_id, []), doc_event_mentions.get(
        target_doc_id, []
    )


def list_available_doc_ids_with_mentions(data_path: str) -> list[str]:
    """データファイル内のmentionsを持つ利用可能なdoc_idのリストを取得する関数

    Args:
        data_path: 入力データのパス（例: "event/ge11_json/dev.json"）

    Returns:
        mentionsを持つ利用可能なdoc_idのリスト

    """
    # entity_mentionsとevent_mentionsを抽出
    doc_entity_mentions, doc_event_mentions = extract_mentions_from_json(data_path)

    # mentionsを持つdoc_idのリストを返す
    return [
        doc_id
        for doc_id in doc_entity_mentions.keys()
        if any(mentions for mentions in doc_entity_mentions[doc_id])
        or any(mentions for mentions in doc_event_mentions[doc_id])
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="event/ge11_json/dev.jsonからmentionsを抽出するスクリプト"
    )
    parser.add_argument(
        "--data_path", default="event/ge11_json/dev.json", help="入力データのパス"
    )
    parser.add_argument(
        "--output_dir", default="event/output/mentions", help="出力ディレクトリのパス"
    )
    parser.add_argument(
        "--list_doc_ids",
        action="store_true",
        help="mentionsを持つ利用可能なdoc_idのリストを表示",
    )
    parser.add_argument("--doc_id", help="特定のdoc_idのmentionsを表示")

    args = parser.parse_args()

    if args.list_doc_ids:
        doc_ids = list_available_doc_ids_with_mentions(args.data_path)
        print(f"mentionsを持つ利用可能なdoc_id ({len(doc_ids)}件):")
        for doc_id in sorted(doc_ids):
            print(f"  - {doc_id}")

    elif args.doc_id:
        entity_mentions, event_mentions = get_mentions_for_doc_id(
            args.data_path, args.doc_id
        )
        print(f"{args.doc_id}のmentionsデータ:")
        print(
            f"  - entity_mentions: {sum(len(mentions) for mentions in entity_mentions)}件"
        )
        print(
            f"  - event_mentions: {sum(len(mentions) for mentions in event_mentions)}件"
        )
        print("\nEntity Mentions:")
        print(json.dumps(entity_mentions, ensure_ascii=False, indent=2))
        print("\nEvent Mentions:")
        print(json.dumps(event_mentions, ensure_ascii=False, indent=2))

    else:
        # デフォルトの動作: すべてのdoc_idごとにファイルを保存
        doc_entity_mentions, doc_event_mentions = save_mentions_by_doc_id(
            args.data_path, args.output_dir
        )
        print(f"doc_idの件数: {len(doc_entity_mentions)}")
        print(
            f"entity_mentionsの合計件数: {sum(sum(len(mentions) for mentions in windows) for windows in doc_entity_mentions.values())}"
        )
        print(
            f"event_mentionsの合計件数: {sum(sum(len(mentions) for mentions in windows) for windows in doc_event_mentions.values())}"
        )
