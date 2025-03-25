import json
import os
import re
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils import postprocess_gold_json

# configの読み込み
# with open("config.json", "r") as f:
#     config = json.load(f)
# ge_or_enzyme = config["ge_or_enzyme"]

ge_or_enzyme = "ge"

if ge_or_enzyme == "ge":
    CORE_ARGS = {
        "Gene_expression": ["Theme"],
        "Transcription": ["Theme"],
        "Protein_catabolism": ["Theme"],
        "Phosphorylation": ["Theme"],
        "Localization": ["Theme"],
        "Binding": ["Theme"],
        "Regulation": ["Theme", "Cause"],
        "Positive_regulation": ["Theme", "Cause"],
        "Negative_regulation": ["Theme", "Cause"],
    }
else:
    CORE_ARGS = {
        "WholeReaction": [],
        "NucleophilicAttack": [],
        "Protonation": [],
        "Deprotonation": [],
        "Stabilisation": [],
        "Destabilisation": [],
        "Activation": [],
        "Inactivation": [],
        "Modulation": [],
        "ElectrophilicAttack": [],
        "Cleavage": [],
        "BondFormation": [],
        "HybridisationChange": [],
        "CouplingReaction": [],
        "UncouplingReaction": [],
        "Others": [],
        "Interaction": [],
        "Release": [],
        "ConformationalChange": [],
    }


def empty_folder(folder_path):
    # フォルダが存在するか確認
    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist.")
        return

    # フォルダ内の全てのファイルとサブディレクトリを削除
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # ファイルまたはシンボリックリンクを削除
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # サブディレクトリを再帰的に削除
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def check_arguments(event_type: str, arguments: list[dict]) -> bool:
    """イベントの引数が正しいかチェックします。"""
    core_args = CORE_ARGS[event_type]
    for core_arg in core_args:
        if core_arg not in arguments:
            return False
    return True


def add_arguments(event_type: str, arguments: list[dict]) -> list[dict]:
    """イベントの引数を補完します。"""
    core_args = CORE_ARGS[event_type]
    arg_role = [arg.split(":")[0] for arg in arguments]
    for core_arg in core_args:
        if core_arg not in arg_role:
            arguments.append(f"{core_arg}:T0")
    return arguments


def json_to_txt_a1_a2(json_file: str, output_dir: str) -> None:
    """JSONデータから元の.txt, .a1, .a2ファイル形式に変換します。"""
    with open(json_file, encoding="utf-8") as file:
        data_entries: list[dict] = [json.loads(line) for line in file]

    # trigger_lines = []  # トリガーの行を格納するリスト
    # event_lines = []  # イベントの行を格納するリスト
    # trigger_set = set()  # トリガーIDを格納して重複チェック用

    empty_folder(output_dir)  # 出力先ディレクトリを空にする

    for i, entry in enumerate(data_entries):
        doc_id = entry["doc_id"]

        # ファイルパスを指定
        txt_file = os.path.join(output_dir, f"{doc_id}.txt")
        a1_file = os.path.join(output_dir, f"{doc_id}.a1")
        a2_file = os.path.join(output_dir, f"{doc_id}.a2")

        # .txtファイルの内容
        # sentence = entry['sentence']
        # with open(txt_file, 'a', encoding='utf-8') as f_txt:
        #     f_txt.write(sentence + '\n')

        Entity_type_list = []
        # .a1ファイル (エンティティ情報)
        entity_mentions = entry["entity_mentions"]
        with open(a1_file, "a", encoding="utf-8") as f_a1:
            for entity in entity_mentions:
                entity_id = entity["id"].split("-")[-1]
                entity_line = f"{entity_id}\t{entity['entity_type']} {entity['start']} {entity['end']}\t{entity['text']}\n"
                f_a1.write(entity_line)
                if entity["entity_type"] == "Entity":
                    Entity_type_list.append(entity_line)

        trigger_lines = []  # トリガーの行を格納するリスト
        event_lines = []  # イベントの行を格納するリスト
        trigger_set = set()  # トリガーIDを格納して重複チェック用

        # .a2ファイル (イベント情報)
        event_mentions = entry.get("event_mentions", [])
        for event in event_mentions:
            trigger = event["trigger"]
            trigger_id = trigger["id"].split("-")[-1]

            # トリガー行がまだ追加されていない場合に追加
            if trigger_id not in trigger_set:
                trigger_line = f"{trigger_id}\t{event['event_type']} {trigger['start']} {trigger['end']}\t{trigger['text']}\n"
                trigger_lines.append(trigger_line)
                trigger_set.add(trigger_id)

            event_id = event["id"].split("-")[-1]
            event_line = f"{event_id}\t{event['event_type']}:{trigger_id} "

            # 引数 (arguments) がある場合の処理
            for argument in event["arguments"]:
                try:
                    argument_id = argument["entity_id"].split("-")[-1]
                except Exception:
                    continue
                # argument_id = argument['entity_id'].split('-')[-1]
                event_line += f"{argument['role']}:{argument_id} "
                # Entityの処理だと思うけど一時的にコメントアウト
                # if argument['role'] == 'Site' or argument['role'] == 'Site2' or argument['role'] == 'CSite' or argument['role'] == 'AtLoc' or argument['role'] == 'ToLoc':
                #     for entity_mention in entity_mentions:
                #         if argument_id in entity_mention['id']:
                #             trigger_line = f"{argument_id}\t{'Entity'} {entity_mention['start']} {entity_mention['end']}\t{entity_mention['text']}\n"
                #             if trigger_line not in trigger_lines:
                #                 trigger_lines.append(trigger_line)
                #                 trigger_set.add(argument_id)
            event_line = re.sub(r"\s$", "", event_line)
            # event_info = event_line.split('\t')[1].split(' ')
            # event_type = event_info[0].split(':')[0]
            # arguments = [arg.split(":")[0] for arg in event_info[1:]]
            # if not  check_arguments(event_type, arguments):
            #     arguments = add_arguments(event_type, event_info[1:])
            #     # print(f"Arguments are added: {event_line} -> {event_info[0]} {' '.join(arguments)}")
            #     event_line = f"{event_id}\t{event_info[0]} {' '.join(arguments)}"
            event_lines.append(event_line + "\n")

        # 重複削除後のトリガー行を書き込み
        trigger_lines = list(dict.fromkeys(trigger_lines))  # 重複を削除
        with open(a2_file, "a", encoding="utf-8") as f_a2:
            f_a2.writelines(trigger_lines)
            f_a2.writelines(event_lines)
            for entity_line in Entity_type_list:
                f_a2.write(entity_line)


if __name__ == "__main__":
    # gold-gold用
    # postprocess_gold_json("ge11_json/dev.json", "converted_output/converted_all.json")
    # 使用例
    json_file = "event/output/converted_all.json"  # JSONファイルのパス
    output_dir = "event/output/reverted_files"  # 出力先ディレクトリ
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    # 変換を実行
    json_to_txt_a1_a2(json_file, output_dir)
