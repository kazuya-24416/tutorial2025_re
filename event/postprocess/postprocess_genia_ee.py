import json
from pathlib import Path
import re
from typing import Any


def delete_invalid_event_mentions(
    event_mentions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Delete invalid event mentions based on missing theme or site arguments.

    Args:
        event_mentions: List of event mentions to process

    Returns:
        list[dict[str, Any]]: List of valid event mentions

    """
    deleted_event_mentions = []
    for i, event_mention in enumerate(event_mentions):
        theme_flag = False
        site_flag = False
        theme2_flag = False
        site2_flag = False
        for argument in event_mention["arguments"]:
            if argument["role"] == "Theme":
                theme_flag = True
            elif argument["role"] == "Site":
                site_flag = True
            elif argument["role"] == "Theme2":
                theme2_flag = True
            elif argument["role"] == "Site2":
                site2_flag = True
        if site_flag and not theme_flag:
            deleted_event_mentions.append(i)
        if site2_flag and not theme2_flag:
            deleted_event_mentions.append(i)

    event_mentions = [
        x for i, x in enumerate(event_mentions) if i not in deleted_event_mentions
    ]
    return event_mentions


def delete_invalid_event_mentions_no_defined(
    event_mentions: list[dict[str, Any]], entity_mentions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Delete invalid event mentions based on undefined entity arguments.

    Args:
        event_mentions: List of event mentions to process
        entity_mentions: List of entity mentions to process

    Returns:
        list[dict[str, Any]]: List of valid event mentions

    """
    max_iter = 100
    iter_count = 0
    while True:
        iter_count += 1
        if iter_count > max_iter:
            break
        undefine_delete = []
        for i, event in enumerate(event_mentions):
            for arg in event["arguments"]:
                arg_id = arg["entity_id"]
                if arg_id is None:
                    undefine_delete.append(i)
                    continue

                # entityとの比較
                entity_match = False
                for entity in entity_mentions:
                    if entity["id"] is not None and arg_id in entity["id"]:
                        entity_match = True
                        break
                # triggerとの比較
                trigger_match = False
                for conv_event in event_mentions:
                    if (
                        conv_event["trigger"]["id"] is not None
                        and arg_id in conv_event["trigger"]["id"]
                    ):
                        trigger_match = True
                        break
                # eventとの比較
                event_match = False
                for conv_event in event_mentions:
                    if conv_event["id"] is not None and arg_id in conv_event["id"]:
                        event_match = True
                        break

                if not entity_match and not trigger_match and not event_match:
                    undefine_delete.append(i)
        # 重複を除去してからソート
        undefine_delete = list(set(undefine_delete))
        undefine_delete.sort(reverse=True)
        # 削除すべきイベントがない場合はループを抜ける
        if not undefine_delete:
            break
        # イベントの削除
        for i in undefine_delete:
            try:
                del event_mentions[i]
            except IndexError:
                pass
    return event_mentions


def convert_descriptions_to_event_mentions(
    sorted_descriptions: list[list[str]],
    wnd_id: str,
    entity_mentions: list[dict[str, Any]] | None = None,
    event_id_counter: int = 1,
    trigger_id_counter: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Convert sorted event descriptions back to event mentions structure.

    Args:
        sorted_descriptions: List of lists of descriptions
        wnd_id: Window ID to prefix event IDs
        entity_mentions: List of entity mentions (optional)
        event_id_counter: Counter for event IDs (継続するために外部から与える)
        trigger_id_counter: Counter for trigger IDs (継続するために外部から与える)

    Returns:
        Tuple containing:
        - List of event mentions in the required format
        - List of entity mentions in the required format
        - Next event_id_counter value
        - Next trigger_id_counter value

    """
    all_event_mentions = []
    all_entity_mentions = []
    entity_id_counter = 1  # entity_idは後で再割り当てされるため初期値は重要ではない

    # entity_mentionsがNoneの場合は空リストに設定
    if entity_mentions is None:
        entity_mentions = []

    # トリガーIDを追跡するための辞書
    trigger_ids = {}  # (trigger_text, event_type) -> trigger_id

    for descriptions_group in sorted_descriptions:
        # トリガーとイベントタイプの組み合わせで追跡
        # Dictionary mapping (trigger, event_type, entity_text) to role for entity arguments
        entity_arguments = {}  # (trigger, event_type, entity_text) -> role
        event_arguments = {}  # (parent_trigger, parent_type, child_trigger, child_type) -> role

        # 各説明文を解析してイベントと引数の関係を構築
        for desc in descriptions_group:
            # メインイベント情報を抽出
            event_match = re.search(
                r"The (.*?) event which is triggered by \"(.*?)\"", desc
            )
            if not event_match:
                continue

            event_type = event_match.group(1)
            trigger = event_match.group(2)

            # エンティティ引数のチェック
            entity_match = re.search(r"has \"(.*?)\" \((.*?)\) as (.*?)\.", desc)
            if entity_match:
                entity_text = entity_match.group(1)
                entity_type = entity_match.group(2)
                role = entity_match.group(3)
                entity_arguments[(trigger, event_type, entity_text)] = role

            # イベント引数のチェック
            event_arg_match = re.search(
                r"has the (.*?) event which is triggered by \"(.*?)\" as (.*?),", desc
            )
            if event_arg_match:
                child_event_type = event_arg_match.group(1)
                child_trigger = event_arg_match.group(2)
                role = event_arg_match.group(3)
                event_arguments[
                    (trigger, event_type, child_trigger, child_event_type)
                ] = role

        # イベントを作成する前に、まず(trigger, event_type)ごとにすべての引数をグループ化
        event_entity_args = {}  # (trigger, event_type) -> list of (entity_text, role)
        event_event_args = {}  # (trigger, event_type) -> list of ((child_trigger, child_type), role)

        for (trigger, event_type, entity_text), role in entity_arguments.items():
            key = (trigger, event_type)
            if key not in event_entity_args:
                event_entity_args[key] = []
            event_entity_args[key].append((entity_text, role))

        for (
            trigger,
            event_type,
            child_trigger,
            child_type,
        ), role in event_arguments.items():
            key = (trigger, event_type)
            child_key = (child_trigger, child_type)
            if key not in event_event_args:
                event_event_args[key] = []
            event_event_args[key].append((child_key, role))

        # 最初にエンティティ引数を持つイベントを作成
        created_events = {}  # (trigger, event_type) -> event_id

        # エンティティ引数を持つイベントを作成
        for key, args in event_entity_args.items():
            trigger, event_type = key
            event_id = f"{wnd_id}-E{event_id_counter}"
            event_id_counter += 1

            # トリガーIDを取得または作成
            trigger_key = (trigger, event_type)
            if trigger_key not in trigger_ids:
                trigger_ids[trigger_key] = f"{wnd_id}-T{trigger_id_counter}"
                trigger_id_counter += 1

            trigger_id = trigger_ids[trigger_key]

            event = {
                "id": event_id,
                "event_type": event_type.capitalize().replace(" ", "_"),
                "trigger": {
                    "id": trigger_id,
                    "text": trigger,
                    "type": event_type.capitalize().replace(" ", "_"),
                },
                "arguments": [],
            }

            # すべてのエンティティ引数を追加
            for entity_text, role in args:
                # entity_mentionsからエンティティIDを取得
                entity_id = next(
                    (em["id"] for em in entity_mentions if em["text"] == entity_text),
                    None,
                )
                if entity_id is None:
                    entity_id = f"{wnd_id}-T{entity_id_counter}"
                    entity_id_counter += 1

                    # エンティティメンションを作成
                    entity_mention = {
                        "id": entity_id,
                        "text": entity_text,
                        "entity_type": "Protein",  # デフォルト値
                        "mention_type": "Protein",  # デフォルト値
                        "entity_subtype": "Protein",  # デフォルト値
                        "start": 0,  # 後で更新される
                        "end": 0,  # 後で更新される
                    }
                    all_entity_mentions.append(entity_mention)

                if role == "csite":
                    role = "CSite"
                elif role == "toloc":
                    role = "ToLoc"
                elif role == "atloc":
                    role = "AtLoc"
                else:
                    role = role.capitalize()
                event["arguments"].append(
                    {"entity_id": entity_id, "text": entity_text, "role": role}
                )

            all_event_mentions.append(event)
            created_events[key] = event_id

        # イベント引数のみを持つイベントを作成
        for key in set(event_event_args.keys()):
            if key not in created_events:
                trigger, event_type = key
                event_id = f"{wnd_id}-E{event_id_counter}"
                event_id_counter += 1

                # トリガーIDを取得または作成
                trigger_key = (trigger, event_type)
                if trigger_key not in trigger_ids:
                    trigger_ids[trigger_key] = f"{wnd_id}-T{trigger_id_counter}"
                    trigger_id_counter += 1

                trigger_id = trigger_ids[trigger_key]

                event = {
                    "id": event_id,
                    "event_type": event_type.capitalize().replace(" ", "_"),
                    "trigger": {
                        "id": trigger_id,
                        "text": trigger,
                        "type": event_type.capitalize().replace(" ", "_"),
                    },
                    "arguments": [],
                }

                all_event_mentions.append(event)
                created_events[key] = event_id

        # 子イベントを先に処理するために、イベントを適切な順序で処理
        # まず、すべての子イベントを作成
        child_events_created = set()

        # すべての子イベントキーを抽出
        all_child_keys = set()
        for args in event_event_args.values():
            for child_key, _ in args:
                all_child_keys.add(child_key)

        # 子イベントを先に作成（エンティティ引数のあるイベントが優先）
        for child_key in all_child_keys:
            if child_key not in created_events and child_key in event_entity_args:
                trigger, event_type = child_key
                event_id = f"{wnd_id}-E{event_id_counter}"
                event_id_counter += 1

                # トリガーIDを取得または作成
                trigger_key = (trigger, event_type)
                if trigger_key not in trigger_ids:
                    trigger_ids[trigger_key] = f"{wnd_id}-T{trigger_id_counter}"
                    trigger_id_counter += 1

                trigger_id = trigger_ids[trigger_key]

                event = {
                    "id": event_id,
                    "event_type": event_type.capitalize().replace(" ", "_"),
                    "trigger": {
                        "id": trigger_id,
                        "text": trigger,
                        "type": event_type.capitalize().replace(" ", "_"),
                    },
                    "arguments": [],
                }

                # エンティティ引数を追加
                for entity_text, role in event_entity_args[child_key]:
                    # entity_mentionsからエンティティIDを取得
                    entity_id = next(
                        (
                            em["id"]
                            for em in entity_mentions
                            if em["text"] == entity_text
                        ),
                        None,
                    )
                    if entity_id is None:
                        entity_id = f"{wnd_id}-T{entity_id_counter}"
                        entity_id_counter += 1

                        # エンティティメンションを作成
                        entity_mention = {
                            "id": entity_id,
                            "text": entity_text,
                            "entity_type": "Protein",  # デフォルト値
                            "mention_type": "Protein",  # デフォルト値
                            "entity_subtype": "Protein",  # デフォルト値
                            "start": 0,  # 後で更新される
                            "end": 0,  # 後で更新される
                        }
                        all_entity_mentions.append(entity_mention)

                    if role == "csite":
                        role = "CSite"
                    elif role == "toloc":
                        role = "ToLoc"
                    elif role == "atloc":
                        role = "AtLoc"
                    else:
                        role = role.capitalize()
                    event["arguments"].append(
                        {"entity_id": entity_id, "text": entity_text, "role": role}
                    )

                all_event_mentions.append(event)
                created_events[child_key] = event_id
                child_events_created.add(child_key)

        # イベント間の関係を構築
        for key, args in event_event_args.items():
            if key not in created_events:
                continue

            parent_event_id = created_events[key]
            parent_event = next(
                e for e in all_event_mentions if e["id"] == parent_event_id
            )

            for child_key, role in args:
                # 子イベントが既に作成されているか確認
                if child_key in created_events:
                    child_event_id = created_events[child_key]
                    child_event = next(
                        e for e in all_event_mentions if e["id"] == child_event_id
                    )

                    # 親イベントに子イベントの引数を追加
                    if role == "csite":
                        role = "CSite"
                    elif role == "toloc":
                        role = "ToLoc"
                    elif role == "atloc":
                        role = "AtLoc"
                    else:
                        role = role.capitalize()
                    parent_event["arguments"].append(
                        {
                            "entity_id": child_event_id,
                            "text": child_event["trigger"]["text"],
                            "role": role,
                        }
                    )
    # 不適格なイベントメンションを削除
    # all_event_mentions = delete_invalid_event_mentions(all_event_mentions)

    # 更新されたカウンター値も返す
    return all_event_mentions, all_entity_mentions, event_id_counter, trigger_id_counter


def search_entity_mention(
    text: str, entity_mentions: list[dict[str, Any]]
) -> str | None:
    """テキストからエンティティIDを検索する関数.

    Args:
        text: 検索するテキスト
        entity_mentions: エンティティメンションのリスト

    Returns:
        見つかった場合はエンティティID、見つからない場合はNone

    """
    if not entity_mentions:
        return None

    # 正確な一致を検索
    for entity_mention in entity_mentions:
        if (
            isinstance(entity_mention, dict)
            and "text" in entity_mention
            and "id" in entity_mention
            and entity_mention["text"] == text
        ):
            return entity_mention["id"]

    # 部分一致を検索（必要に応じて）
    for entity_mention in entity_mentions:
        if (
            isinstance(entity_mention, dict)
            and "text" in entity_mention
            and "id" in entity_mention
            and (text in entity_mention["text"] or entity_mention["text"] in text)
        ):
            return entity_mention["id"]

    return None


def relabel_entity_id(
    converted_event_mentions: list[dict[str, Any]],
    entity_mentions_by_window: list[list[dict[str, Any]]],
) -> None:
    """イベントメンションのエンティティIDを再ラベル付けする関数.

    Args:
        converted_event_mentions: 変換されたイベントメンションのリスト
        entity_mentions_by_window: ウィンドウごとのエンティティメンションのリスト

    """
    # ウィンドウIDをキーとするエンティティメンションの辞書を作成
    entity_mentions_dict = {}
    for window_entities in entity_mentions_by_window:
        if not window_entities:
            continue
        # ウィンドウIDを取得（最初のエンティティから）
        if isinstance(window_entities[0], dict) and "id" in window_entities[0]:
            window_id = "-".join(window_entities[0]["id"].split("-")[:-1])
            # このウィンドウIDのエンティティメンションを辞書に登録
            entity_mentions_dict[window_id] = window_entities

    # 各イベントメンションのIDを置き換え
    for event_mention in converted_event_mentions:
        # イベントのウィンドウIDを取得
        event_window_id = "-".join(event_mention["id"].split("-")[:-1])
        # このウィンドウのエンティティメンションを取得
        window_entities = entity_mentions_dict.get(event_window_id, [])
        # 引数のエンティティIDを置き換え
        for argument in event_mention["arguments"]:
            entity_id = search_entity_mention(argument["text"], window_entities)
            if entity_id:
                argument["entity_id"] = entity_id


def relabel_trigger_position(
    converted_event_mentions: list[dict[str, Any]],
    event_mentions_by_window: list[list[dict[str, Any]]],
) -> None:
    """イベントメンションのトリガー位置を再ラベル付けする関数.

    Args:
        converted_event_mentions: 変換されたイベントメンションのリスト
        event_mentions_by_window: ウィンドウごとのイベントメンションのリスト

    """
    # ウィンドウIDをキーとするイベントメンションの辞書を作成
    event_mentions_dict = {}
    for window_events in event_mentions_by_window:
        if not window_events:
            continue
        # ウィンドウIDを取得（最初のエンティティから）
        window_id = "-".join(window_events[0]["id"].split("-")[:-1])
        # このウィンドウIDのイベントメンションを辞書に登録
        event_mentions_dict[window_id] = window_events
    for converted_event_mention in converted_event_mentions:
        event_window_id = "-".join(converted_event_mention["id"].split("-")[:-1])
        window_events = event_mentions_dict.get(event_window_id, [])
        converted_trigger_text = converted_event_mention["trigger"]["text"]
        for window_event in window_events:
            if window_event["trigger"]["text"] == converted_trigger_text:
                converted_event_mention["trigger"]["start"] = window_event["trigger"][
                    "start"
                ]
                converted_event_mention["trigger"]["end"] = window_event["trigger"][
                    "end"
                ]
                break


# ドキュメントIDを抽出するヘルパー関数
def get_doc_id(wnd_id: str) -> str:
    """ウィンドウIDからドキュメントIDを抽出する関数.

    Args:
        wnd_id: ウィンドウID（例: "PMC-1942070-04-Results-01-0"）

    Returns:
        ドキュメントID（例: "PMC-1942070-04-Results-01"）

    """  # noqa: RUF002
    # 最後のハイフン以前の部分をドキュメントIDとする
    parts = wnd_id.split("-")
    return "-".join(parts[:-1])


def process_multiple_windows(
    all_descriptions: list[list[list[str]]],
    all_wnd_ids: list[str],
    entity_mentions_by_window: list[list[dict[str, Any]]],
    event_mentions_by_window: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """複数のウィンドウの説明文を処理し、イベントメンションとエンティティメンションを生成する.

    Args:
        all_descriptions: 全ウィンドウの説明文のリスト
        all_wnd_ids: 全ウィンドウIDのリスト
        entity_mentions_by_window: ウィンドウごとのエンティティメンションのリスト
        event_mentions_by_window: ウィンドウごとのイベントメンションのリスト

    Returns:
        Tuple containing:
        - List of event mentions in the required format
        - List of entity mentions by window in the required format

    """
    # IDのカウンターを初期化
    event_id_counter = 1
    trigger_id_counter = 1000

    all_converted_events = []

    # 入力のエンティティメンションをコピーして使用
    processed_entity_mentions_by_window = []
    if entity_mentions_by_window:
        processed_entity_mentions_by_window = [
            window[:] for window in entity_mentions_by_window
        ]
    else:
        processed_entity_mentions_by_window = [[] for _ in range(len(all_wnd_ids))]

    # 各ウィンドウに対して処理を実行
    for i, (descriptions, wnd_id) in enumerate(
        zip(all_descriptions, all_wnd_ids, strict=False)
    ):
        # 現在のウィンドウのエンティティメンションを取得
        window_entity_mentions = (
            processed_entity_mentions_by_window[i]
            if i < len(processed_entity_mentions_by_window)
            else []
        )

        # 同じドキュメントIDが含まれる場合、カウンターを継続する
        if i > 0 and get_doc_id(wnd_id) == get_doc_id(all_wnd_ids[i - 1]):
            # 前のウィンドウから継続
            converted_events, entity_mentions, event_id_counter, trigger_id_counter = (
                convert_descriptions_to_event_mentions(
                    descriptions,
                    wnd_id,
                    window_entity_mentions,
                    event_id_counter,
                    trigger_id_counter,
                )
            )
        else:
            # 新しいドキュメントの場合はカウンターをリセット
            converted_events, entity_mentions, event_id_counter, trigger_id_counter = (
                convert_descriptions_to_event_mentions(
                    descriptions, wnd_id, window_entity_mentions, 1, 1000
                )
            )
        all_converted_events.extend(converted_events)
        # 新しく生成されたエンティティメンションを追加
        if i < len(processed_entity_mentions_by_window):
            # 重複を避けるために既存のエンティティIDをチェック
            existing_ids = {em["id"] for em in processed_entity_mentions_by_window[i]}
            for em in entity_mentions:
                if em["id"] not in existing_ids:
                    processed_entity_mentions_by_window[i].append(em)
                    existing_ids.add(em["id"])
        else:
            processed_entity_mentions_by_window.append(entity_mentions)
    # エンティティメンションの処理
    relabel_entity_id(all_converted_events, processed_entity_mentions_by_window)
    # イベントメンションの処理
    if event_mentions_by_window:
        relabel_trigger_position(all_converted_events, event_mentions_by_window)
    return all_converted_events, processed_entity_mentions_by_window


def postprocess_genia_ee(
    input_data: str | list[dict[str, Any]] | dict[str, Any],
    entity_mentions_by_window: str | list[list[dict[str, Any]]] | None = None,
    event_mentions_by_window: str | list[list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """メイン処理関数.

    Args:
        input_data: 入力データ（ファイルパス、JSONテキスト、または辞書/リスト）
        entity_mentions_by_window: エンティティメンションのデータ（ファイルパスまたはリスト）
        event_mentions_by_window: イベントメンションのデータ（ファイルパスまたはリスト）

    Returns:
        Tuple containing:
        - List of event mentions in the required format
        - List of entity mentions by window in the required format

    """
    # 入力データの形式に応じた処理
    if isinstance(input_data, str):
        # 文字列の場合: JSONとして解析またはファイルパスとして読み込む
        if Path(input_data).is_file():
            # ファイルパスとして扱う
            with Path(input_data).open(encoding="utf-8") as f:
                file_content = f.read()
            try:
                data = json.loads(file_content)
            except json.JSONDecodeError:
                try:
                    data = json.loads(f"[{file_content}]")
                except json.JSONDecodeError:
                    raise ValueError(
                        f"ファイル '{input_data}' が有効なJSON形式ではありません"
                    )
        else:
            # JSONテキストとして扱う
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                try:
                    data = json.loads(f"[{input_data}]")
                except json.JSONDecodeError:
                    raise ValueError("入力データが有効なJSON形式ではありません")
    elif isinstance(input_data, dict):
        # 単一のデータ辞書の場合はリストに変換
        data = [input_data]
    else:
        # リストまたはその他の場合はそのまま使用
        data = input_data

    # エンティティメンションの処理（ファイルパスの場合は読み込む）
    if isinstance(entity_mentions_by_window, str):
        if Path(entity_mentions_by_window).is_file():
            with Path(entity_mentions_by_window).open(encoding="utf-8") as f:
                entity_data = json.load(f)
            # JSONの形式に応じて処理
            if isinstance(entity_data, list) and all(
                isinstance(item, list) for item in entity_data
            ):
                # 既にウィンドウごとのリスト形式になっている場合
                entity_mentions_by_window = entity_data
            elif isinstance(entity_data, list) and all(
                isinstance(item, dict) for item in entity_data
            ):
                # 辞書のリスト形式の場合、ウィンドウごとにグループ化
                # ウィンドウIDでグループ化
                entity_mentions_by_window = []
                for item in data:
                    wnd_id = item["wnd_id"]
                    window_entities = [
                        e for e in entity_data if e.get("id", "").startswith(wnd_id)
                    ]
                    entity_mentions_by_window.append(window_entities)
            else:
                # その他の形式の場合はそのまま使用
                entity_mentions_by_window = entity_data
        else:
            # ファイルでない場合はJSONとして解析
            try:
                entity_data = json.loads(entity_mentions_by_window)
                entity_mentions_by_window = entity_data
            except json.JSONDecodeError:
                raise ValueError(
                    "エンティティメンションデータが有効なJSON形式ではありません"
                )

    # イベントメンションの処理（ファイルパスの場合は読み込む）
    if (
        isinstance(event_mentions_by_window, str)
        and Path(event_mentions_by_window).is_file()
    ):
        with Path(event_mentions_by_window).open(encoding="utf-8") as f:
            event_data = json.load(f)
        event_mentions_by_window = event_data

    # 各ウィンドウのdescriptionsとwnd_idを抽出
    all_descriptions = []
    all_wnd_ids = []

    for item in data:
        wnd_id = item["wnd_id"]
        output = item["output"]

        # descriptionをグループ化（空行で区切る）
        groups = output.split("\n\n")

        # 各グループの説明を行ごとに分割
        descriptions = [group.split("\n") for group in groups]

        all_descriptions.append(descriptions)
        all_wnd_ids.append(wnd_id)

    # エンティティメンションがない場合は、ウィンドウの数だけ空リストを作成
    if entity_mentions_by_window is None:
        entity_mentions_by_window = [[] for _ in range(len(all_wnd_ids))]

    # 複数ウィンドウの処理を実行
    converted_events, entity_mentions = process_multiple_windows(
        all_descriptions,
        all_wnd_ids,
        entity_mentions_by_window,
        event_mentions_by_window,
    )

    return converted_events, entity_mentions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="イベントメンション生成スクリプト")
    parser.add_argument("input_file", help="入力JSONファイル")
    parser.add_argument("--entity_file", help="エンティティメンションJSONファイル")
    parser.add_argument("--event_file", help="イベントメンションJSONファイル")
    parser.add_argument(
        "--output_file",
        help="出力イベントJSONファイル（デフォルト: event_mentions.json）",
        default="event_mentions.json",
    )
    parser.add_argument(
        "--output_entity_file",
        help="出力エンティティJSONファイル（デフォルト: entity_mentions.json）",
        default="entity_mentions.json",
    )

    args = parser.parse_args()

    # メイン関数を実行
    event_mentions, entity_mentions = postprocess_genia_ee(
        args.input_file, args.entity_file, args.event_file
    )

    # イベントメンションを保存
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(event_mentions, f, indent=2, ensure_ascii=False)

    # エンティティメンションを保存
    with open(args.output_entity_file, "w", encoding="utf-8") as f:
        json.dump(entity_mentions, f, indent=2, ensure_ascii=False)

    print(
        f"イベントメンションを {args.output_file} に保存しました（{len(event_mentions)} 件）"
    )
    print(
        f"エンティティメンションを {args.output_entity_file} に保存しました（{sum(len(entities) for entities in entity_mentions)} 件）"
    )
