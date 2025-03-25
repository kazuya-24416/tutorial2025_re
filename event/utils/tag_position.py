import copy
import json

# from utils import calculate_all_spans


def calculate_all_spans(sentence: str, entity_mentions: list) -> list[tuple[int, int]]:
    """Calculate spans for all words in the sentence relative to a reference word's position.

    Args:
        sentence (str): The input sentence
        reference_word (str): The reference word
        reference_span (tuple): The span (start, end) of the reference word in the sentence

    Returns:
        dict: Dictionary mapping each word to its (start, end) span

    """
    for entity in entity_mentions:
        # raise ValueError(f"Could not find '{reference_word}' in the sentence")
        reference_word = entity["text"]
        reference_span = (entity["start"], entity["end"])
        ref_pos = sentence.find(reference_word)
        if ref_pos != -1:
            break
    if ref_pos == -1:
        return []

    sentence_start = reference_span[0] - ref_pos

    # Split sentence into words and track their positions
    spans = []
    current_pos = 0

    # Iterate through each word while keeping track of its position
    words = sentence.split()
    for word in words:
        # Find the word's position in the original sentence starting from current_pos
        word_pos = sentence.find(word, current_pos)
        if word_pos == -1:
            continue

        # Calculate absolute spans
        word_start = sentence_start + word_pos
        word_end = word_start + len(word)

        # Store in dictionary
        spans.append([word_start, word_end])

        # Update current position to look for next word after this one
        current_pos = word_pos + len(word)

    return spans


def tag_occurrence_sentence(text, sentence_start=0):
    """文章中の単語やシンボルにタグを付ける

    Parameters
    ----------
    text (str): タグを付ける文章
    sentence_start (int): 文章の開始位置

    Returns
    -------
    tuple: (タグ付き文章, 位置情報の辞書, スペース情報の辞書)

    """
    position_map = {}  # (start, end) -> tagged_token
    space_info = {}  # (start, end) -> {"before": bool, "after": bool}
    occurrence_count = {}  # token -> count

    tokens = []
    current_pos = 0

    while current_pos < len(text):
        # スペースをスキップ
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
        if current_pos >= len(text):
            break

        # 記号を検出
        if any(text[current_pos] == c for c in "-/_.,;:()[]{}!?"):
            token_start = current_pos
            token = text[current_pos]
            current_pos += 1
            token_end = current_pos
            tokens.append((token, token_start, token_end))
        else:
            # 通常の単語を検出
            token_start = current_pos
            while (
                current_pos < len(text)
                and not text[current_pos].isspace()
                and not any(text[current_pos] == c for c in "-/_.,;:()[]{}!?")
            ):
                current_pos += 1
            token = text[token_start:current_pos]
            token_end = current_pos
            tokens.append((token, token_start, token_end))

    # スペース情報を収集とタグ付きテキストの生成
    tagged_tokens = []
    for i, (token, start, end) in enumerate(tokens):
        # 出現回数を更新
        occurrence_count[token] = occurrence_count.get(token, 0) + 1

        # 前のスペースを確認
        has_space_before = False
        if i > 0:
            prev_end = tokens[i - 1][2]
            between_text = text[prev_end:start]
            has_space_before = " " in between_text

        # 後ろのスペースを確認
        has_space_after = False
        if i < len(tokens) - 1:
            next_start = tokens[i + 1][1]
            between_text = text[end:next_start]
            has_space_after = " " in between_text

        # スペース情報を保存
        space_info[(start + sentence_start, end + sentence_start)] = {
            "before": has_space_before,
            "after": has_space_after,
        }

        # タグ付きトークンを作成
        tagged_token = f"{token}^{{{occurrence_count[token]}}}"
        position_map[(start + sentence_start, end + sentence_start)] = tagged_token

        # タグ付きテキストの生成用
        if has_space_before:
            tagged_tokens.append(" ")
        tagged_tokens.append(tagged_token)

    # タグ付きテキストを生成
    tagged_text = "".join(tagged_tokens)

    return tagged_text, position_map, space_info


# def tag_occurrence_sentence(text, sentence_start=0):
#     """
#     文章中の単語やシンボルにタグを付ける

#     Parameters:
#     text (str): タグを付ける文章
#     sentence_start (int): 文章の開始位置

#     Returns:
#     tuple: (タグ付き文章, 位置情報の辞書, スペース情報の辞書)
#     """
#     position_map = {}  # (start, end) -> tagged_token
#     space_info = {}   # (start, end) -> {"before": bool, "after": bool}

#     tokens = []
#     current_pos = 0

#     while current_pos < len(text):
#         # スペースをスキップ
#         while current_pos < len(text) and text[current_pos].isspace():
#             current_pos += 1
#         if current_pos >= len(text):
#             break

#         # 記号を検出
#         if any(text[current_pos] == c for c in '-/_.,;:()[]{}!?'):
#             token_start = current_pos
#             token = text[current_pos]
#             current_pos += 1
#             token_end = current_pos
#             tokens.append((token, token_start, token_end))
#         else:
#             # 通常の単語を検出
#             token_start = current_pos
#             while current_pos < len(text) and not text[current_pos].isspace() and not any(text[current_pos] == c for c in '-/_.,;:()[]{}!?'):
#                 current_pos += 1
#             token = text[token_start:current_pos]
#             token_end = current_pos
#             tokens.append((token, token_start, token_end))

#     # スペース情報を収集とタグ付きテキストの生成
#     tagged_tokens = []
#     for i, (token, start, end) in enumerate(tokens):
#         # 前のスペースを確認
#         has_space_before = False
#         if i > 0:
#             prev_end = tokens[i-1][2]
#             between_text = text[prev_end:start]
#             has_space_before = ' ' in between_text

#         # 後ろのスペースを確認
#         has_space_after = False
#         if i < len(tokens) - 1:
#             next_start = tokens[i+1][1]
#             between_text = text[end:next_start]
#             has_space_after = ' ' in between_text

#         # スペース情報を保存
#         space_info[(start + sentence_start, end + sentence_start)] = {
#             "before": has_space_before,
#             "after": has_space_after
#         }

#         # タグ付きトークンを作成
#         tagged_token = f"{token}^{{1}}"
#         position_map[(start + sentence_start, end + sentence_start)] = tagged_token

#         # タグ付きテキストの生成用
#         if has_space_before:
#             tagged_tokens.append(' ')
#         tagged_tokens.append(tagged_token)

#     # タグ付きテキストを生成
#     tagged_text = ''.join(tagged_tokens)

#     return tagged_text, position_map, space_info


def get_tagged_text_from_position_map(
    position_map, space_info, start, end, original_text
):
    """position_mapから指定された範囲のタグ付きテキストを取得する
    Parameters:
    position_map (dict): 位置情報の辞書
    space_info (dict): スペース情報の辞書
    start (int): 開始位置
    end (int): 終了位置
    original_text (str): 元のテキスト
    Returns:
    str: タグ付きテキスト
    """
    # 範囲内のトークンを取得
    tagged_tokens = []
    sorted_positions = sorted(
        position_map.items(), key=lambda x: x[0][0]
    )  # 開始位置でソート

    # 範囲内のトークンを収集
    for (token_start, token_end), tagged_token in sorted_positions:
        if (start <= token_start < end) or (token_start <= start < token_end):
            tagged_tokens.append((tagged_token, token_start, token_end))

    # トークンをマージ
    result = []
    for i, (token, token_start, token_end) in enumerate(tagged_tokens):
        # 前のトークンとの間のスペースを確認
        if i > 0:
            # スペース情報を使用
            prev_space_info = space_info.get((token_start, token_end), {})
            if prev_space_info.get("before", False):
                result.append(" ")
        result.append(token)

    return "".join(result)


def create_tagged_entity_mentions(sentence, sentence_start, entity_mentions):
    """エンティティメンションにタグを付与する関数

    Args:
        sentence (str): 対象となるテキスト全体
        sentence_start (int): 文章の開始位置
        entity_mentions (list): エンティティメンションのリスト

    Returns:
        list: タグ付けされたエンティティメンションのリスト

    """
    if not entity_mentions:
        return []

    # 文章をタグ付けし、位置情報を取得
    tagged_text, position_map, space_info = tag_occurrence_sentence(
        sentence, sentence_start
    )

    # 入力リストのディープコピーを作成
    tagged_mentions = copy.deepcopy(entity_mentions)

    # 各メンションのtextフィールドを更新
    for mention in tagged_mentions:
        # タグ付きのテキストを取得
        tagged_text = get_tagged_text_from_position_map(
            position_map, space_info, mention["start"], mention["end"], mention["text"]
        )
        # textフィールドのみを更新
        mention["text"] = tagged_text

    return tagged_mentions


def create_tagged_event_mentions(
    sentence, sentence_start, event_mentions, tagged_entity_mentions
):
    """イベントメンションにタグを付与する関数

    Args:
        sentence (str): 対象となるテキスト全体
        sentence_start (int): 文章の開始位置
        event_mentions (list): イベントメンションのリスト
        tagged_entity_mentions (list): タグ付けされたエンティティメンションのリスト

    Returns:
        list: タグ付けされたイベントメンションのリスト

    """
    if not event_mentions:
        return []

    # 文章をタグ付けし、位置情報を取得
    tagged_text, position_map, space_info = tag_occurrence_sentence(
        sentence, sentence_start
    )

    # 入力リストのディープコピーを作成
    tagged_mentions = copy.deepcopy(event_mentions)

    # エンティティIDとタグ付きテキストのマッピングを作成
    entity_text_map = {
        entity["id"]: entity["text"] for entity in tagged_entity_mentions
    }

    # 各イベントメンションのトリガーと引数のtextを更新
    for mention in tagged_mentions:
        # トリガーのタグ付きテキストを取得
        trigger_text = get_tagged_text_from_position_map(
            position_map,
            space_info,
            mention["trigger"]["start"],
            mention["trigger"]["end"],
            mention["trigger"]["text"],
        )
        # トリガーのtextフィールドのみを更新
        mention["trigger"]["text"] = trigger_text

        # 引数のtextフィールドを更新
        for arg in mention["arguments"]:
            # エンティティの場合
            if arg["entity_id"].split("-")[-1].startswith("T"):
                arg["text"] = entity_text_map.get(arg["entity_id"])

    # イベント引数のテキストを更新
    event_text_map = {
        event["id"]: event["trigger"]["text"] for event in tagged_mentions
    }
    for event in tagged_mentions:
        for arg in event["arguments"]:
            if (
                not arg["entity_id"].split("-")[-1].startswith("T")
            ):  # イベントの引数の場合
                arg["text"] = event_text_map.get(arg["entity_id"])

    return tagged_mentions


# if __name__ == "__main__":
#     with open("ge11_json/dev.json", "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             sentence_info = json.loads(line)
#             tagged_text, position_map, space_info = tag_occurrence_sentence(sentence_info["sentence"], sentence_info["sentence_start"])
#             print(tagged_text)

#             # エンティティメンションの作成とチェック
#             tagged_entity_mentions = create_tagged_entity_mentions(sentence_info["sentence"], sentence_info["sentence_start"], sentence_info["entity_mentions"])
#             for entity, tag_entity in zip(sentence_info["entity_mentions"], tagged_entity_mentions):
#                 print("entity:", entity["text"], "->", tag_entity["text"])
#                 # タグ付きテキストが文章に含まれているか確認
#                 if tag_entity["text"] == "serine^{1}82^{1}" or tag_entity["text"] == "Interleukin^{1}10^{1}":
#                     continue
#                 assert tag_entity["text"] in tagged_text, f"Tagged entity text '{tag_entity['text']}' not found in tagged sentence"
#             print()

#             # イベントメンションの作成とチェック
#             tagged_event_mentions = create_tagged_event_mentions(sentence_info["sentence"], sentence_info["sentence_start"], sentence_info["event_mentions"], tagged_entity_mentions)
#             for event, tag_event in zip(sentence_info["event_mentions"], tagged_event_mentions):
#                 print("trigger:", event["trigger"]["text"], "->", tag_event["trigger"]["text"])
#                 # トリガーのタグ付きテキストが文章に含まれているか確認
#                 assert tag_event["trigger"]["text"] in tagged_text, f"Tagged trigger text '{tag_event['trigger']['text']}' not found in tagged sentence"
#                 for arg, tag_arg in zip(event["arguments"], tag_event["arguments"]):
#                     print("arg:", arg["text"], "->", tag_arg["text"])
#                     # 引数のタグ付きテキストが文章に含まれているか確認
#                     if tag_arg["text"]:  # tag_arg["text"]がNoneでない場合のみチェック
#                         if tag_entity["text"] == "serine^{1}82^{1}" or tag_entity["text"] == "Interleukin^{1}10^{1}":
#                             continue
#                         assert tag_arg["text"] in tagged_text, f"Tagged argument text '{tag_arg['text']}' not found in tagged sentence"


# import json
# import copy

# def calculate_all_spans(sentence: str, entity_mentions: list) -> list[tuple[int, int]]:
#     """
#     Calculate spans for all words in the sentence relative to a reference word's position.

#     Args:
#         sentence (str): The input sentence
#         reference_word (str): The reference word
#         reference_span (tuple): The span (start, end) of the reference word in the sentence

#     Returns:
#         dict: Dictionary mapping each word to its (start, end) span
#     """

#     for entity in entity_mentions:
#         # raise ValueError(f"Could not find '{reference_word}' in the sentence")
#         reference_word = entity["text"]
#         reference_span = (entity["start"], entity["end"])
#         ref_pos = sentence.find(reference_word)
#         if ref_pos != -1:
#             break
#     if ref_pos == -1:
#         return []

#     sentence_start = reference_span[0] - ref_pos

#     # Split sentence into words and track their positions
#     spans = []
#     current_pos = 0

#     # Iterate through each word while keeping track of its position
#     words = sentence.split()
#     for word in words:
#         # Find the word's position in the original sentence starting from current_pos
#         word_pos = sentence.find(word, current_pos)
#         if word_pos == -1:
#             continue

#         # Calculate absolute spans
#         word_start = sentence_start + word_pos
#         word_end = word_start + len(word)

#         # Store in dictionary
#         spans.append([word_start, word_end])

#         # Update current position to look for next word after this one
#         current_pos = word_pos + len(word)

#     return spans

# def tag_occurrence_sentence(text):
#     """
#     文章中の各単語の出現順序を記録する関数
#     ピリオドやその他の句読点を単語から分離して処理

#     Parameters:
#     text (str): 分析対象の文章

#     Returns:
#     dict: 各単語の出現位置を示す辞書
#     list: 元の文章の各単語に出現順序を付けたリスト
#     """
#     # 単語ごとの出現回数を記録する辞書
#     word_count = {}
#     # 単語ごとの出現順序を記録する辞書
#     word_positions = {}
#     # 結果を格納するリスト
#     result = []

#     # 文章を単語に分割
#     words = text.split()

#     for word in words:
#         # ピリオドがある場合は分離
#         base_word = word.rstrip('.,')
#         has_period = word.endswith('.')
#         has_comma = word.endswith(',')

#         # 現在の単語の出現回数をカウント
#         if base_word in word_count:
#             word_count[base_word] += 1
#         else:
#             word_count[base_word] = 1

#         # 現在の出現順序を記録
#         current_occurrence = word_count[base_word]

#         # 単語と出現順序の組み合わせを結果リストに追加
#         # ピリオドがあった場合は元の位置に戻す
#         result_word = f"{base_word}^{current_occurrence}"
#         if has_period:
#             result_word += "."
#         if has_comma:
#             result_word += ","
#         result.append(result_word)

#         # 単語の全ての出現位置を記録
#         if base_word not in word_positions:
#             word_positions[base_word] = []
#         word_positions[base_word].append(len(result))

#     return " ".join(result)

# def copy_entity_without_text(entity_mention):
#     # 元の辞書からコピーを作成し、'text'キーを削除
#     new_entity = entity_mention.copy()
#     new_entity.pop('text', None)
#     return new_entity

# def tag_occurrecne_entity_mentions(sentence, entity_mentions):
#     # タグづけされた文章を元に、entity_mentionsのtextを変更する
#     tagged_sentence = tag_occurrence_sentence(sentence)

#     # sentence内の単語とpositionの対応を取得
#     if not entity_mentions:
#         return []
#     word_split = sentence.split(" ")
#     tagged_word_split = tagged_sentence.split(" ")
#     span = calculate_all_spans(sentence, entity_mentions)
#     if len(span) == 0:
#         return []
#     tag_entity_mentions = [copy_entity_without_text(entity) for entity in entity_mentions]

#     del_list = []
#     for i, (entity, tag_entity) in enumerate(zip(entity_mentions, tag_entity_mentions)):
#         entity_text = entity["text"]
#         entity_span = (entity["start"], entity["end"])
#         entity_word_count = len(entity_text.split(" "))

#         # エンティティの開始位置を含む単語を探す
#         matching_words = []
#         start_idx = None
#         for j, (word, tagged_word, word_span) in enumerate(zip(word_split, tagged_word_split, span)):
#             if word_span[0] <= entity_span[0] <= word_span[1] or entity_span[0] <= word_span[0] <= word_span[1]:
#                 start_idx = j
#                 break

#         if start_idx is not None:
#             # エンティティの単語数分だけ連続する単語を取得
#             matching_tagged_words = tagged_word_split[start_idx:start_idx + entity_word_count]
#             if len(matching_tagged_words) == entity_word_count:
#                 tag_entity['text'] = ' '.join(matching_tagged_words)
#             else:
#                 del_list.append(i)
#         else:
#             del_list.append(i)

#         #最後にピリオドがついている場合は削除
#         if tag_entity.get('text') is not None:
#             if tag_entity['text'][-1] == ".":
#                 tag_entity['text'] = tag_entity['text'][:-1]

#     # 見つからなかったエンティティを削除
#     tag_entity_mentions = [x for i, x in enumerate(tag_entity_mentions) if i not in del_list]

#     for entity in tag_entity_mentions:
#         entity_text = entity["text"]
#         assert entity_text in tagged_sentence
#         assert entity_text[-1] != "."

#     return tag_entity_mentions


# def copy_event_without_text(event_mention):
#     """
#     event_mentionsのtriggerとargumentsのtextを削除したコピーを作成する関数

#     Parameters:
#     event_mention (dict): 元のevent_mentionデータ

#     Returns:
#     dict: triggerとargumentsのtextを除いた新しい辞書
#     """
#     # 深いコピーを作成して元のデータを保護
#     new_event = event_mention.copy()

#     # triggerが存在する場合、そのコピーを作成してtextを削除
#     if 'trigger' in new_event:
#         new_event['trigger'] = new_event['trigger'].copy()
#         new_event['trigger'].pop('text', None)

#     # argumentsが存在する場合の処理
#     if 'arguments' in new_event:
#         new_event['arguments'] = [arg.copy() for arg in event_mention['arguments']]
#         for arg in new_event['arguments']:
#             arg.pop('text', None)

#     return new_event

# def extract_text_by_id(entities, target_id):
#     # 与えられたエンティティリストから指定されたIDに一致するテキストを抽出する
#     for entity in entities:
#         if entity['id'] == target_id:
#             return entity['text']
#     return None

# def tag_occurrence_event_mentions(sentence, event_mentions, entity_mentions):
#     if not event_mentions:
#         return []
#     if not entity_mentions:
#         return []
#     # タグづけされた文章, entity_mentionを取得
#     tagged_sentence = tag_occurrence_sentence(sentence)
#     tag_entity_mentions = tag_occurrecne_entity_mentions(sentence, entity_mentions)

#     # sentence内の単語とpositionの対応を取得
#     word_split = sentence.split(" ")
#     tagged_word_split = tagged_sentence.split(" ")
#     span = calculate_all_spans(sentence, entity_mentions)

#     # 返すEvent Mentionのリスト
#     tag_event_mentions = [copy_event_without_text(event) for event in event_mentions]

#     # triggerのtextを変更
#     del_list = []
#     for i, event in enumerate(event_mentions):
#         trigger = event["trigger"]
#         trigger_text = trigger["text"]
#         trigger_span = (trigger["start"], trigger["end"])
#         trigger_word_count = len(trigger_text.split(" "))

#         # triggerの開始位置を含む単語を探す
#         start_idx = None
#         for j, (word, tagged_word, word_span) in enumerate(zip(word_split, tagged_word_split, span)):
#             if word_span[0] <= trigger_span[0] <= word_span[1] or trigger_span[0] <= word_span[0] <= word_span[1]:
#                 start_idx = j
#                 break

#         if start_idx is not None:
#             # triggerの単語数分だけ連続する単語を取得
#             matching_tagged_words = tagged_word_split[start_idx:start_idx + trigger_word_count]
#             if len(matching_tagged_words) == trigger_word_count:
#                 tag_event_mentions[i]["trigger"]["text"] = ' '.join(matching_tagged_words)
#             else:
#                 del_list.append(i)
#         else:
#             del_list.append(i)

#         #最後にピリオドがついている場合は削除
#         if "text" in tag_event_mentions[i]["trigger"]:
#             if tag_event_mentions[i]["trigger"]["text"][-1] == ".":
#                 tag_event_mentions[i]["trigger"]["text"] = tag_event_mentions[i]["trigger"]["text"][:-1]

#     # 見つからなかったイベントを削除
#     tag_event_mentions = [x for i, x in enumerate(tag_event_mentions) if i not in del_list]

#     del_arg_list = []
#     # argumentsのtextを変更
#     for i, event in enumerate(tag_event_mentions):
#         for arg in event["arguments"]:
#             # argumentがentityの場合
#             if arg["entity_id"].split("-")[-1][0] == "T":
#                 arg_entity_id = arg["entity_id"]
#                 tag_arg_text = extract_text_by_id(tag_entity_mentions, arg_entity_id)
#                 for tag_arg in tag_event_mentions[i]["arguments"]:
#                     if tag_arg["entity_id"] == arg_entity_id:
#                         tag_arg["text"] = tag_arg_text
#                         break
#             # argumentがeventの場合
#             else:
#                 for tag_arg in tag_event_mentions[i]["arguments"]:
#                     for tag_event in tag_event_mentions:
#                         if tag_event["id"] == tag_arg["entity_id"]:
#                             tag_arg_text = tag_event["trigger"]["text"]
#                             tag_arg["text"] = tag_arg_text
#                             break
#         for tag_arg in tag_event_mentions[i]["arguments"]:
#             if "text" not in tag_arg or tag_arg["text"] == None:
#                 del_arg_list.append(i)

#         # 最後にピリオドがついている場合は削除
#         for tag_arg in tag_event_mentions[i]["arguments"]:
#             if "text" in tag_arg and tag_arg["text"] is not None:
#                 if tag_arg["text"][-1] == ".":
#                     tag_arg["text"] = tag_arg["text"][:-1]

#     tag_event_mentions = [x for i, x in enumerate(tag_event_mentions) if i not in del_arg_list]

#     for event in tag_event_mentions:
#         trigger = event["trigger"]["text"]
#         assert trigger in tagged_sentence
#         assert trigger[-1] != "."
#         for arg in event["arguments"]:
#             arg_text = arg["text"]
#             assert arg_text in tagged_sentence
#             assert arg_text[-1] != "."

#     return tag_event_mentions

# def tag_word_with_count(sentence: str, sentence_start: int, word: str, word_span: tuple[int, int]) -> str:
#     """
#     文章中の単語に対して、同じ単語が出現した順番でタグを付与する関数

#     Args:
#         sentence (str): 対象となるテキスト全体
#         sentence_start (int): 文章の開始位置
#         word (str): タグ付けする単語
#         word_span (tuple): 単語の開始位置と終了位置のタプル (start, end)

#     Returns:
#         str: タグ付けされた単語
#     """
#     # 文章から実際のテキストを抽出して確認
#     start, end = word_span
#     actual_text = sentence[start-sentence_start:end-sentence_start]

#     # 抽出したテキストと指定された単語が一致するか確認
#     if actual_text != word:
#         raise ValueError(f"Text mismatch at position {start}-{end}. "
#                         f"Expected: {word}, Found: {actual_text}")

#     # 文章内でその単語が何回目に出現するかをカウント
#     current_pos = 0
#     occurrence_count = 0

#     while current_pos < len(sentence):
#         # 単語を探す
#         found_pos = sentence.find(word, current_pos)
#         if found_pos == -1:
#             break

#         occurrence_count += 1

#         # 現在チェックしている単語の範囲に達したら終了
#         if found_pos == start - sentence_start:
#             break

#         current_pos = found_pos + 1

#     # タグ付きの単語を返す
#     return f"{word}^{occurrence_count}"

# def create_tagged_entity_mentions(sentence: str, sentence_start: int, entity_mentions: list) -> list:
#     """
#     エンティティメンションにタグを付与する関数

#     Args:
#         sentence (str): 対象となるテキスト全体
#         sentence_start (int): 文章の開始位置
#         entity_mentions (list): エンティティメンションのリスト

#     Returns:
#         list: タグ付けされたエンティティメンションのリスト
#     """
#     if not entity_mentions:
#         return []

#     # 結果を格納するリスト
#     tagged_mentions = []

#     # start位置でソート
#     mentions = sorted(copy.deepcopy(entity_mentions), key=lambda x: x['start'])

#     for mention in mentions:
#         # タグ付きのテキストを取得
#         tagged_text = tag_word_with_count(
#             sentence,
#             sentence_start,
#             mention['text'],
#             (mention['start'], mention['end'])
#         )

#         # 新しいメンションオブジェクトを作成
#         tagged_mention = {
#             'id': mention['id'],
#             'entity_type': mention['entity_type'],
#             'mention_type': mention['mention_type'],
#             'entity_subtype': mention['entity_subtype'],
#             'start': mention['start'],
#             'end': mention['end'],
#             'text': tagged_text
#         }

#         tagged_mentions.append(tagged_mention)

#     return tagged_mentions

# def create_tagged_event_mentions(sentence: str, sentence_start: int, event_mentions: list, tagged_entity_mentions: list) -> list:
#     """
#     イベントメンションにタグを付与する関数

#     Args:
#         sentence (str): 対象となるテキスト全体
#         sentence_start (int): 文章の開始位置
#         event_mentions (list): イベントメンションのリスト
#         tagged_entity_mentions (list): タグ付けされたエンティティメンションのリスト

#     Returns:
#         list: タグ付けされたイベントメンションのリスト
#     """
#     if not event_mentions:
#         return []

#     # 結果を格納するリスト
#     tagged_mentions = []

#     for mention in event_mentions:
#         # ディープコピーを作成
#         mention_copy = copy.deepcopy(mention)

#         # タグ付きのトリガーを取得
#         trigger_text = tag_word_with_count(
#             sentence,
#             sentence_start,
#             mention_copy['trigger']['text'],
#             (mention_copy['trigger']['start'], mention_copy['trigger']['end'])
#         )
#         # 新しいトリガーオブジェクトを作成
#         tagged_trigger = {
#             'start': mention_copy['trigger']['start'],
#             'end': mention_copy['trigger']['end'],
#             'text': trigger_text
#         }

#         # 新しいイベントオブジェクトを作成
#         tagged_mention = {
#             'id': mention_copy['id'],
#             'event_type': mention_copy['event_type'],
#             'arguments': mention_copy['arguments'],
#             'trigger': tagged_trigger
#         }

#         tagged_mentions.append(tagged_mention)

#     # 引数のtextを変更
#     for mention in tagged_mentions:
#         for arg in mention['arguments']:
#             # エンティティの場合
#             if arg['entity_id'].split("-")[-1].startswith('T'):
#                 entity_text = next((entity['text'] for entity in tagged_entity_mentions if entity['id'] == arg['entity_id']), None)
#                 arg['text'] = entity_text
#             # イベントの場合
#             else:
#                 event_trigger = next((event['trigger']['text'] for event in tagged_mentions if event['id'] == arg['entity_id']), None)
#                 arg['text'] = event_trigger

#     return tagged_mentions

if __name__ == "__main__":
    with open("ge11_json/dev.json", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sentence_info = json.loads(line)
            tagged_entity_mentions = create_tagged_entity_mentions(
                sentence_info["sentence"],
                sentence_info["sentence_start"],
                sentence_info["entity_mentions"],
            )
            tagged_sentence, _, _ = tag_occurrence_sentence(
                sentence_info["sentence"], sentence_info["sentence_start"]
            )
            tagged_event_mentions = create_tagged_event_mentions(
                sentence_info["sentence"],
                sentence_info["sentence_start"],
                sentence_info["event_mentions"],
                tagged_entity_mentions,
            )
            print(tagged_sentence)
            print(tagged_entity_mentions)
            print(tagged_event_mentions)
