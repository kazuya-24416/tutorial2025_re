import json
import re
from pathlib import Path


def add_spans_to_jsonl(input_file: str, output_file: str) -> None:
    """JSONLinesファイルのne_listにspanアノテーションをリスト形式で追加する.

    - 同じエンティティが複数回登場する場合、start/endリストに全ての出現位置を記録
    - 部分一致の問題を回避するため、単語境界を考慮する

    Args:
        input_file (str): 入力JSONLinesファイルのパス
        output_file (str): 出力ファイルのパス

    """
    processed_lines = []

    with Path(input_file).open(encoding="utf-8") as f:
        for line in f:
            example = json.loads(line.strip())
            text = example["text"]

            # 各エンティティに対してリスト形式のspanを追加
            for entity in example["ne_list"]:
                entity_text = entity["entity"]

                # 単語境界を考慮した正規表現パターンを作成
                # パターン：(文頭|非日本語文字)(エンティティ)(文末|非日本語文字)
                pattern = (
                    r"(^|[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])("
                    + re.escape(entity_text)
                    + r")([^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]|$)"
                )

                # 全ての出現を見つける
                matches = list(re.finditer(pattern, text))

                # spanリストを初期化
                start_positions = []
                end_positions = []

                if matches:
                    # 各出現位置を記録
                    for match in matches:
                        # マッチグループ2がエンティティ本体
                        start = match.start(2)
                        end = match.end(2)
                        start_positions.append(start)
                        end_positions.append(end)
                else:
                    # 部分一致で検索（正規表現が見つからない場合のフォールバック）
                    idx = 0
                    while idx < len(text):
                        start = text.find(entity_text, idx)
                        if start == -1:
                            break

                        end = start + len(entity_text)
                        idx = end  # 次の検索開始位置を更新

                        # 単語境界をチェック
                        is_boundary_start = start == 0 or not is_japanese_char(
                            text[start - 1]
                        )
                        is_boundary_end = (
                            end == len(text) or not is_japanese_char(text[end])
                            if end < len(text)
                            else True
                        )

                        if is_boundary_start and is_boundary_end:
                            start_positions.append(start)
                            end_positions.append(end)

                # どうしても見つからない場合
                if not start_positions:
                    # 単語境界を無視して検索
                    matches = list(re.finditer(re.escape(entity_text), text))
                    if matches:
                        for match in matches:
                            start_positions.append(match.start())
                            end_positions.append(match.end())
                    else:
                        # どうしても見つからない場合は空リストを設定
                        print(
                            f"警告: テキスト「{text}」中にエンティティ「{entity_text}」が見つかりません"
                        )

                # リスト形式のspanを追加
                entity["start"] = start_positions
                entity["end"] = end_positions

            processed_lines.append(example)

    # 処理したデータを出力
    with open(output_file, "w", encoding="utf-8") as f:
        for example in processed_lines:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(
        f"{len(processed_lines)}件のデータにリスト形式のspanアノテーションを追加し、{output_file}に書き込みました。"
    )


def is_japanese_char(char: str) -> bool:
    """文字が日本語（ひらがな、カタカナ、漢字）かどうかをチェック."""
    if not char:
        return False
    # ひらがな、カタカナ、漢字の文字コード範囲
    return (
        "\u3040" <= char <= "\u309f"  # ひらがな
        or "\u30a0" <= char <= "\u30ff"  # カタカナ
        or "\u4e00" <= char <= "\u9fff"
    )  # 漢字


# 使用例
if __name__ == "__main__":
    add_spans_to_jsonl("data/debug.jsonl", "data/debug_with_list_spans.jsonl")
