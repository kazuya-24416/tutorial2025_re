import re
import subprocess
import sys
from pathlib import Path

import argparse
import json

from .create_doc_input_data import get_input_data_for_doc_id
from .extract_mentions import (
    get_mentions_for_doc_id,
    get_sentence_for_doc_id,
    get_sentence_start_for_doc_id,
)
from .postprocess_genia_ee import (
    delete_invalid_event_mentions,
    delete_invalid_event_mentions_no_defined,
    postprocess_genia_ee,
)
from .postprocess_json2a2 import json_to_txt_a1_a2
from ..utils.tag_position import (
    create_tagged_entity_mentions,
    create_tagged_event_mentions,
)

DEV_DATA_EE_PATH = "LLaMA-Factory/data/dev_genia_ee.json"
GE11_DEV_JSON_PATH = "event/ge11_json/dev.json"


def run_evaluation(
    ref_dir: str,
    pred_dir: str,
    *,
    softboundary: bool = False,
    partialrecursive: bool = False,
    singlepenalty: bool = False,
    verifytext: bool = False,
    spliteval: bool = False,
    verbose: bool = False,
) -> dict[str, float]:
    """Run the evaluation by directly calling the original eval_ev_ge.py script via subprocess.

    and extracting the results using regular expressions.

    Args:
        ref_dir: Directory containing reference files
        pred_dir: Directory containing prediction files
        softboundary: Whether to use soft boundary matching
        partialrecursive: Whether to use partial recursive matching of event arguments
        singlepenalty: Whether to use single penalty for partial matches
        verifytext: Whether to require that correct texts are given for textbound annotations
        spliteval: Whether to evaluate in "split events" mode (relaxes constraints)
        verbose: Whether to print verbose output

    Returns:
        dict: Dictionary containing evaluation scores for each event type

    """  # noqa: E501
    # Construct the command to run the original evaluation script
    eval_script_path = (
        Path(__file__).parent / ".." / "evaluation" / "eval_ev_ge.py"
    ).resolve()

    cmd = [sys.executable, str(eval_script_path)]

    # Add reference directory
    cmd.append("-r")
    cmd.append(ref_dir)

    # Add prediction directory
    cmd.append("-d")
    cmd.append(pred_dir)

    # Add options
    if softboundary:
        cmd.append("-s")
    if partialrecursive:
        cmd.append("-p")
    if singlepenalty:
        cmd.append("-1")
    if verifytext:
        cmd.append("-t")
    if spliteval:
        cmd.append("-S")
    if verbose:
        cmd.append("-v")

    # Run the command and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        stderr = result.stderr

        if stderr and verbose:
            print(stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script: {e}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        return {}

    # Parse the results using regular expressions
    scores = {}

    # Define regex patterns for extracting event types results
    event_pattern = r"(\s*[=]*\[*\w+[_\-]*\w*\]*[=]*)\s+(\d+) \(\s*(\d+)\)\s+(\d+) \(\s*(\d+)\)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"  # noqa: E501

    # Extract individual event types
    for line in output.split("\n"):
        match = re.search(event_pattern, line.strip())
        if match:
            event_type = match.group(1).strip()
            gold_count = int(match.group(2))
            gold_match = int(match.group(3))
            answer_count = int(match.group(4))
            answer_match = int(match.group(5))
            recall = float(match.group(6))
            precision = float(match.group(7))
            fscore = float(match.group(8))

            # Clean up event type (remove extra spaces, equals signs and brackets)
            event_type = (
                event_type.strip().replace("=", "").replace("[", "").replace("]", "")
            )

            # Map special event types to their clean names
            if "SVT-TOTAL" in event_type:
                event_type = "SVT-TOTAL"
            elif "EVT-TOTAL" in event_type:
                event_type = "EVT-TOTAL"
            elif "REG-TOTAL" in event_type:
                event_type = "REG-TOTAL"
            elif "TOTAL" in event_type:
                event_type = "TOTAL"

            scores[event_type] = {
                "gold_count": gold_count,
                "gold_match": gold_match,
                "answer_count": answer_count,
                "answer_match": answer_match,
                "recall": recall,
                "precision": precision,
                "fscore": fscore,
            }

    # Print the original output
    print(output)

    return scores


def get_doc_id_from_json(data_path: str) -> list[str]:
    """Get a list of doc_ids from a JSON file.

    Args:
        data_path: Path to the JSON file

    Returns:
        list[str]: List of doc_ids

    """
    doc_ids = []
    # データを読み込む
    with open(data_path, encoding="utf-8") as f:
        # Check first character to determine if it's a JSON array or JSON Lines
        first_char = f.read(1)
        f.seek(0)  # Reset file pointer
        # Use ternary operator to simplify the code
        data = json.load(f) if first_char == "[" else [json.loads(line) for line in f]
    for item in data:
        if item["doc_id"] in doc_ids:
            continue
        doc_ids.append(item["doc_id"])
    return doc_ids


def process_single_doc_id(
    doc_id: str,
    data_path: str = DEV_DATA_EE_PATH,
    entity_mentions: list[dict[str, str]] | None = None,
    event_mentions: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    """Process a single doc_id and return converted events and entities.

    Args:
        doc_id: Document ID
        data_path: Path to the input data
        entity_mentions: Optional entity mentions
        event_mentions: Optional event mentions

    Returns:
        Optional tuple of converted events and entities

    """
    input_data = get_input_data_for_doc_id(data_path, doc_id)

    if not input_data:
        return ([], [])

    # データを処理
    converted_events, converted_entities = postprocess_genia_ee(
        input_data, entity_mentions, event_mentions
    )
    return converted_events, converted_entities


def process_doc_with_mentions(
    doc_id: str, data_path: str = GE11_DEV_JSON_PATH
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    """Process a single doc_id and return converted events and entities.

    Args:
        doc_id: Document ID
        data_path: Path to the input data

    Returns:
        Optional tuple of converted events and entities

    """
    entity_mentions, event_mentions = get_mentions_for_doc_id(data_path, doc_id)
    sentences = get_sentence_for_doc_id(data_path, doc_id)
    sentence_starts = get_sentence_start_for_doc_id(data_path, doc_id)
    assert len(sentences) == len(sentence_starts), (
        f"sentences and sentence_starts must have the same length. doc_id: {doc_id}, len(sentences): {len(sentences)}, len(sentence_starts): {len(sentence_starts)}"  # noqa: E501
    )
    # タグを付ける
    tagged_entity_mentions = []
    tagged_event_mentions = []
    for entity_mention, event_mention, sentence, sentence_start in zip(
        entity_mentions, event_mentions, sentences, sentence_starts, strict=False
    ):
        tagged_entity_mention = create_tagged_entity_mentions(
            sentence, sentence_start, entity_mention
        )
        tagged_event_mention = create_tagged_event_mentions(
            sentence, sentence_start, event_mention, tagged_entity_mention
        )
        tagged_entity_mentions.append(tagged_entity_mention)
        tagged_event_mentions.append(tagged_event_mention)

    if not tagged_entity_mentions and not tagged_event_mentions:
        print(f"Error: doc_id '{doc_id}' not found or has no mentions in {data_path}")
        return None

    return tagged_entity_mentions, tagged_event_mentions


def convert_and_save_data(
    ge11_json_path: str = GE11_DEV_JSON_PATH,
    dev_data_ee_path: str = DEV_DATA_EE_PATH,
    output_json_path: str = "event/output/converted_all.json",
    output_dir: str = "event/output/reverted_files",
) -> None:
    """Convert and save data.

    Args:
        ge11_json_path: GE11のJSONファイルパス
        dev_data_ee_path: 開発データのEEファイルパス
        output_json_path: 出力JSONファイルパス
        output_dir: 出力ディレクトリパス

    """
    # 後処理
    converted_all_info = []
    doc_ids = get_doc_id_from_json(ge11_json_path)
    for doc_id in doc_ids:
        tagged_sentences = get_sentence_for_doc_id(ge11_json_path, doc_id)
        tagged_entity_mentions, tagged_event_mentions = process_doc_with_mentions(
            doc_id, data_path=ge11_json_path
        )
        converted_event, converted_entity = process_single_doc_id(
            doc_id,
            data_path=dev_data_ee_path,
            entity_mentions=tagged_entity_mentions,
            event_mentions=tagged_event_mentions,
        )
        if not converted_entity:
            converted_entity = [[] for _ in tagged_sentences]
        converted_events = [[] for _ in tagged_sentences]
        for converted_ev in converted_event:
            if not converted_ev:
                continue
            wnd_id = converted_ev["id"].split("-")[-2]
            converted_events[int(wnd_id)].append(converted_ev)
        assert len(tagged_sentences) == len(converted_entity) == len(converted_events), f"tagged_sentences: {len(tagged_sentences)} converted_entities: {len(converted_entity)} converted_events: {len(converted_events)}"
        for i, (event, entity, sentence) in enumerate(
            zip(converted_events, converted_entity, tagged_sentences, strict=False)
        ):
            event = delete_invalid_event_mentions(event)
            event = delete_invalid_event_mentions_no_defined(event, entity)
            converted_all_info.append(
                {
                    "doc_id": doc_id,
                    "wnd_id": i,
                    "sentence": sentence,
                    "event_mentions": event,
                    "entity_mentions": entity,
                }
            )
    # jsonに変換
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_json_path).open("w", encoding="utf-8") as f:
        for info in converted_all_info:
            json.dump(info, f)
            f.write("\n")
    # .txt, .a1, .a2に変換
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_to_txt_a1_a2(output_json_path, output_dir)

    return converted_all_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run event extraction evaluation")
    parser.add_argument(
        "-r",
        "--ref_directory",
        required=True,
        help="Directory containing reference files",
    )
    parser.add_argument(
        "-d",
        "--pred_directory",
        required=True,
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "-s", "--softboundary", action="store_true", help="Use soft boundary matching"
    )
    parser.add_argument(
        "-p",
        "--partialrecursive",
        action="store_true",
        help="Use partial recursive matching of event arguments",
    )
    parser.add_argument(
        "-1",
        "--singlepenalty",
        action="store_true",
        help="Use single penalty for partial matches",
    )
    parser.add_argument(
        "-vt",
        "--verifytext",
        action="store_true",
        help="Require that correct texts are given for textbound annotations",
    )
    parser.add_argument(
        "-se",
        "--spliteval",
        action="store_true",
        help="Evaluate in 'split events' mode (relaxes constraints)",
    )
    parser.add_argument(
        "-c",
        "--custom_output",
        action="store_true",
        help="Print additional custom output format",
    )

    args = parser.parse_args()

    # データの変換と保存を実行
    convert_and_save_data(
        GE11_DEV_JSON_PATH,
        DEV_DATA_EE_PATH,
        "event/output/converted_all.json",
        "event/output/reverted_files",
    )

    scores = run_evaluation(
        args.ref_directory,
        args.pred_directory,
        softboundary=args.softboundary,
        partialrecursive=args.partialrecursive,
        singlepenalty=args.singlepenalty,
        verifytext=args.verifytext,
        spliteval=args.spliteval,
        verbose=args.verbose,
    )

    # Print custom output format if requested
    if args.custom_output:
        print("\n評価結果:")
        # Define the order of event types to display
        event_types = [
            "Gene_expression",
            "Transcription",
            "Protein_catabolism",
            "Phosphorylation",
            "Localization",
            "SVT-TOTAL",
            "Binding",
            "EVT-TOTAL",
            "Regulation",
            "Positive_regulation",
            "Negative_regulation",
            "REG-TOTAL",
            "TOTAL",
        ]

        for event_type in event_types:
            if event_type in scores:
                score = scores[event_type]
                print(
                    f"{event_type}: Precision={score['precision']:.2f}, Recall={score['recall']:.2f}, F-score={score['fscore']:.2f}"
                )
