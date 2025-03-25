import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import json

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use("Agg")

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QAPair:
    question: str
    answer: str
    context: dict[str, Any]
    is_valid: bool = True


class GENIAEvent:
    """GENIAのイベントを管理するクラス"""

    def __init__(self, event_data: dict[str, Any]):
        self.id = event_data["id"]
        self.event_type = event_data["event_type"]
        self.trigger = event_data["trigger"]
        self.arguments = event_data["arguments"]

    @property
    def trigger_id(self) -> str:
        """トリガーのIDを取得"""
        return self.trigger["id"]

    @property
    def trigger_text(self) -> str:
        """トリガーのテキストを取得"""
        return self.trigger["text"]

    def get_argument_events(self) -> list[dict[str, Any]]:
        """イベントを引数に取るものを抽出"""
        return [arg for arg in self.arguments if arg.get("arg_event_type") is not None]

    def get_entity_arguments(self) -> list[dict[str, Any]]:
        """エンティティの引数を抽出"""
        return [arg for arg in self.arguments if arg.get("arg_event_type") is None]


class GENIAEventGraph:
    """イベントの集合を管理し、明示的なグラフ構造を構築するクラス"""

    def __init__(self, events_data: list[dict[str, Any]]):
        # イベントをマージしてから初期化
        merged_events = self._merge_duplicate_events(events_data)

        # マージされたイベントにIDを付与（オリジナルのIDが失われている可能性があるため）
        for i, event in enumerate(merged_events):
            event["id"] = f"E{i}"

        self.events = [GENIAEvent(event) for event in merged_events]
        self.event_map = {event.id: event for event in self.events}
        self.graph = nx.DiGraph()
        self._build_graph()

    def _get_event_key(self, event: dict) -> tuple[str, str]:
        """イベントの一意性を判断するためのキーを生成
        異なるタイプを持つ同じトリガーテキストのイベントは別々に扱う

        Args:
            event (Dict): イベントデータ

        Returns:
            Tuple[str, str]: (trigger_text + event_type)のタプル

        """
        # トリガーテキストとイベントタイプを組み合わせてユニークなキーを作成
        return (
            f"{event['trigger']['text']}_{event['event_type']}",
            event["event_type"],
        )

    def _build_graph(self) -> None:
        """イベントとエンティティからグラフを構築"""
        for event in self.events:
            # イベントノードのユニークなIDを生成
            event_node_id = f"{event.trigger_text}_{event.event_type}"  # トリガーとタイプを組み合わせた一意のID

            # イベントノードを追加
            self.graph.add_node(
                event_node_id,
                node_type="event",
                text=event.trigger_text,
                event_type=event.event_type,
            )

            # エンティティ引数のノードとエッジを追加
            for arg in event.get_entity_arguments():
                text = arg["text"]
                if not self.graph.has_node(text):
                    self.graph.add_node(text, node_type="entity", text=text)
                self.graph.add_edge(
                    event_node_id,  # 更新されたイベントノードID
                    text,
                    edge_type="entity_arg",
                    role=arg["role"],
                )

            # イベント引数のエッジを追加
            for arg in event.get_argument_events():
                arg_node_id = f"{arg['text']}_{arg.get('arg_event_type', 'Unknown')}"  # 引数イベントの一意のID
                # イベント引数のノードが存在しない場合は追加
                if not self.graph.has_node(arg_node_id):
                    self.graph.add_node(
                        arg_node_id,
                        node_type="event",
                        text=arg["text"],
                        event_type=arg.get("arg_event_type", "Unknown"),
                    )
                self.graph.add_edge(
                    event_node_id,  # 更新されたイベントノードID
                    arg_node_id,  # 更新された引数ノードID
                    edge_type="event_arg",
                    role=arg["role"],
                )

    def _merge_duplicate_events(self, events: list[dict]) -> list[dict]:
        """同じtrigger textとevent_typeの組み合わせを持つイベントをマージ
        異なるタイプを持つ同じトリガーテキストのイベントは別々に保持

        Args:
            events (List[Dict]): イベントのリスト

        Returns:
            List[Dict]: マージされたイベントのリスト

        """
        event_map = {}

        for event in events:
            key = self._get_event_key(event)

            if key in event_map:
                # 既存のイベントに引数を追加
                event_map[key]["arguments"].extend(event["arguments"])
            else:
                # 新しいイベントとして追加
                event_map[key] = {
                    "trigger": event["trigger"].copy(),
                    "event_type": event["event_type"],
                    "arguments": event["arguments"].copy(),
                }

        # 重複する引数を除去
        for event in event_map.values():
            unique_args = {}
            for arg in event["arguments"]:
                arg_key = (arg["text"], arg["role"])
                if arg_key not in unique_args:
                    unique_args[arg_key] = arg.copy()

            event["arguments"] = list(unique_args.values())

        return list(event_map.values())

    def get_nodes(self) -> list[dict[str, Any]]:
        """ノード情報を取得"""
        return [
            {
                "id": node_id,
                "node_type": self.graph.nodes[node_id]["node_type"],
                "text": self.graph.nodes[node_id]["text"],
                **(
                    {"event_type": self.graph.nodes[node_id]["event_type"]}
                    if self.graph.nodes[node_id]["node_type"] == "event"
                    else {}
                ),
            }
            for node_id in self.graph.nodes
        ]

    def get_edges(self) -> list[dict[str, Any]]:
        """エッジ情報を取得"""
        return [
            {
                "source": src,
                "target": dst,
                "edge_type": self.graph.edges[src, dst]["edge_type"],
                "role": self.graph.edges[src, dst]["role"],
            }
            for src, dst in self.graph.edges
        ]

    def get_event_nodes(self) -> list[dict[str, Any]]:
        """イベントノードのみを取得"""
        return [node for node in self.get_nodes() if node["node_type"] == "event"]

    def get_entity_nodes(self) -> list[dict[str, Any]]:
        """エンティティノードのみを取得"""
        return [node for node in self.get_nodes() if node["node_type"] == "entity"]

    def get_event_arguments(
        self, event_text: str, event_type: str
    ) -> list[dict[str, Any]]:
        """指定されたイベントの引数を取得"""
        event_node_id = f"{event_text}_{event_type}"
        return [
            {
                "target": dst,
                "edge_type": self.graph.edges[event_node_id, dst]["edge_type"],
                "role": self.graph.edges[event_node_id, dst]["role"],
                "text": self.graph.nodes[dst]["text"],
            }
            for dst in self.graph.successors(event_node_id)
        ]

    def get_argument_events(
        self, event_text: str, event_type: str
    ) -> list[dict[str, Any]]:
        """指定されたイベントのイベント引数を取得"""
        return [
            arg
            for arg in self.get_event_arguments(event_text, event_type)
            if arg["edge_type"] == "event_arg"
        ]

    def get_entity_arguments(
        self, event_text: str, event_type: str
    ) -> list[dict[str, Any]]:
        """指定されたイベントのエンティティ引数を取得"""
        return [
            arg
            for arg in self.get_event_arguments(event_text, event_type)
            if arg["edge_type"] == "entity_arg"
        ]

    def is_argument_event(self, event_id: str) -> bool:
        """指定されたイベントが他のイベントの引数として使用されているか確認"""
        return any(
            dst == event_id and self.graph.edges[src, dst]["edge_type"] == "event_arg"
            for src, dst in self.graph.in_edges(event_id)
        )

    def is_leaf_entity(self, entity_id: str) -> bool:
        """エンティティが葉ノードかどうかを確認"""
        return (
            self.graph.has_node(entity_id)
            and self.graph.nodes[entity_id]["node_type"] == "entity"
            and self.graph.out_degree(entity_id) == 0
        )

    def get_graph_structure(self) -> dict[str, Any]:
        """グラフ構造全体を取得"""
        return {"nodes": self.get_nodes(), "edges": self.get_edges()}

    def to_networkx(self) -> nx.DiGraph:
        """NetworkXのグラフとして取得"""
        return self.graph.copy()

    def _calculate_node_levels(self) -> dict[str, int]:
        """各ノードの階層レベルを計算"""
        levels = {}
        # ルートノード（入次数が0）を見つける
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        # 幅優先探索で階層を割り当て
        visited = set()
        current_level = roots
        level_num = 0

        while current_level:
            next_level = []
            for node in current_level:
                if node not in visited:
                    levels[node] = level_num
                    visited.add(node)
                    next_level.extend(
                        [n for n in self.graph.successors(node) if n not in visited]
                    )
            current_level = list(set(next_level))  # 重複を除去
            level_num += 1

        return levels

    def _create_hierarchical_layout(self, levels: dict[str, int]) -> dict[str, tuple]:
        """階層レベルに基づいてノードの位置を計算"""
        pos = {}
        if not levels:  # レベルが空の場合
            return pos

        # 各レベルのノードを収集
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        # 各レベルのノードを配置
        max_level = max(levels.values())
        for level in range(max_level + 1):
            nodes = level_nodes.get(level, [])
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                # x座標は水平方向に均等に配置
                x = (i - (n_nodes - 1) / 2) / max(n_nodes, 1)
                # y座標は階層に応じて上から下に配置（逆転させる）
                y = (max_level - level) / max(max_level, 1)
                pos[node] = (x, y)

        return pos

    def visualize(self, figsize=(12, 8)) -> plt.Figure | None:
        """グラフを階層的に可視化"""
        if not self.graph.nodes():  # グラフが空の場合
            print("Warning: Empty graph, nothing to visualize")
            return None

        fig = plt.figure(figsize=figsize)

        # ノードの階層レベルを計算
        levels = self._calculate_node_levels()
        if not levels:  # レベルが空の場合
            print("Warning: No hierarchy levels found in the graph")
            plt.close(fig)
            return None

        # 階層的なレイアウトを作成
        pos = self._create_hierarchical_layout(levels)
        if not pos:  # レイアウトが空の場合
            print("Warning: Could not create layout for the graph")
            plt.close(fig)
            return None

        # ノードの色分けとサイズ設定
        event_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "event"
        ]
        entity_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "entity"
        ]

        # イベントノードの描画
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=event_nodes,
            node_color="lightgreen",
            node_size=3000,
            edgecolors="black",
            linewidths=1,
        )

        # エンティティノードの描画
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=entity_nodes,
            node_color="lightblue",
            node_size=2000,
            edgecolors="black",
            linewidths=1,
        )

        # エッジの描画
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color="black",
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.2",
            width=1.5,
        )

        # ノードラベルの描画
        labels = {
            n: f"{d['text']}\n({d.get('event_type', 'Entity')})"
            for n, d in self.graph.nodes(data=True)
        }
        label_pos = {n: (p[0], p[1] - 0.05) for n, p in pos.items()}
        nx.draw_networkx_labels(
            self.graph, label_pos, labels=labels, font_size=10, font_weight="bold"
        )

        # エッジラベルの描画
        edge_labels = nx.get_edge_attributes(self.graph, "role")
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=9,
            label_pos=0.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )

        plt.title("GENIA Event Graph")
        plt.margins(0.2)
        plt.axis("off")
        plt.tight_layout()

        return fig

    def to_str(self, fmt="text"):
        if fmt == "text":
            # Use the original text representation logic
            # (Assuming there is a method or code block for text output; otherwise, inline the original __str__ code here)
            return self._to_text()

        if fmt == "mermaid":
            # Use the existing mermaid representation logic
            return self._to_mermaid()

        if fmt == "graphviz":
            # New Graphviz (DOT format) representation
            return self._to_graphviz()

        if fmt == "hrt":
            # New HRT format representation
            return self._to_hrt()

        raise ValueError(f"Unknown format: {fmt}")

    def __str__(self):
        # Default to text representation
        return self.to_str("mermaid")

    def _to_text(self):
        output = []
        output.append("Events:")
        for event in self.events:
            output.append(f"  - Type: {event.event_type}")
            output.append(f"  - Trigger: {event.trigger_text}")

            # エンティティ引数を出力
            entity_args = self.get_entity_arguments(
                event.trigger_text, event.event_type
            )
            if entity_args:
                output.append("  - Entity Arguments:")
                for arg in entity_args:
                    output.append(f"    * {arg['role']}: {arg['text']}")

            # イベント引数を出力
            event_args = self.get_argument_events(event.trigger_text, event.event_type)
            if event_args:
                output.append("  - Event Arguments:")
                for arg in event_args:
                    output.append(f"    * {arg['role']}: {arg['text']}")

            output.append("")  # 空行を追加して読みやすく

        return "\n".join(output)

    def _to_mermaid(self):
        output = ["graph TD;"]

        # ノードの定義
        node_ids = {}  # node_key -> id のマッピング
        for idx, node in enumerate(self.get_nodes()):
            node_id = f"n{idx}"
            # ノードの一意キーを生成
            if node["node_type"] == "event":
                node_key = f"{node['text']}_{node['event_type']}"
            else:
                node_key = node["text"]
            node_ids[node_key] = node_id

            # ノードの形状とラベルを設定
            if node["node_type"] == "event":
                # イベントノードは角丸四角形
                output.append(f"{node_id}[{node['text']}<br>({node['event_type']})]")
            else:
                # エンティティノードは楕円形
                output.append(f"{node_id}(({node['text']}))")

        # エッジの定義
        for edge in self.get_edges():
            # ソースノードの一意キーを生成
            src_node = self.graph.nodes[edge["source"]]
            if src_node["node_type"] == "event":
                src_key = f"{src_node['text']}_{src_node['event_type']}"
            else:
                src_key = edge["source"]

            # ターゲットノードの一意キーを生成
            dst_node = self.graph.nodes[edge["target"]]
            if dst_node["node_type"] == "event":
                dst_key = f"{dst_node['text']}_{dst_node['event_type']}"
            else:
                dst_key = edge["target"]

            src_id = node_ids[src_key]
            dst_id = node_ids[dst_key]
            output.append(f"{src_id} -->|{edge['role']}| {dst_id}")

        return "\n".join(output)

    def _to_graphviz(self):
        dot_lines = ["digraph G {"]
        for node in self.get_nodes():
            dot_lines.append(
                f'  "{node["id"]}" [label="{node["text"]}\n({node.get("event_type", "Entity")})"];'
            )
        for edge in self.get_edges():
            dot_lines.append(
                f'  "{edge["source"]}" -> "{edge["target"]}" [label="{edge["role"]}"];'
            )
        dot_lines.append("}")
        return "\n".join(dot_lines)

    def _to_hrt(self):
        """Return a string representation of the graph in hrt format.
        Each edge is represented as:
        'head is <head>. relation is <role>. tail is <tail>.'
        """
        lines = []
        for edge in self.get_edges():
            head_node = self.graph.nodes[edge["source"]]
            tail_node = self.graph.nodes[edge["target"]]
            line = f"head is {head_node['text']} ({head_node.get('event_type', 'Entity')}). relation is {edge['role']}. tail is {tail_node['text']}."
            lines.append(line)
        return "\n".join(lines)


class EventGraphReconstructor:
    """質問応答からイベントグラフを再構築するクラス"""

    def __init__(self):
        self.events = []
        self.current_event = None

    def process_qa_pair(self, qa: QAPair):
        """質問応答ペアを処理してグラフ情報を抽出"""
        if qa.question.startswith("What are events of"):
            self._process_event_question(qa)
        elif qa.question.startswith("What are arguments of"):
            self._process_argument_question(qa)

    def _process_event_question(self, qa: QAPair):
        """イベントに関する質問を処理"""
        if not qa.is_valid or qa.answer == "No events found.":
            return

        # 回答を個別のイベント情報に分割
        event_statements = qa.answer.split(". Event trigger")

        for statement in event_statements:
            if "trigger is" not in statement and "type is" not in statement:
                continue

            try:
                # トリガーの抽出
                trigger = None
                if "trigger is" in statement:
                    trigger_parts = statement.split("trigger is ")
                    if len(trigger_parts) > 1:
                        trigger = trigger_parts[1].split(".")[0].strip()

                # イベントタイプの抽出
                event_type = None
                if "type is" in statement:
                    type_parts = statement.split("type is ")
                    if len(type_parts) > 1:
                        event_type = type_parts[1].split(".")[0].strip()

                # 有効なイベント情報が得られた場合のみ追加
                if trigger and event_type:
                    self.current_event = {
                        "trigger": {"text": trigger},
                        "event_type": event_type,
                        "arguments": [],
                    }
                    self.events.append(self.current_event)

            except (IndexError, AttributeError):
                print(f"Warning: Failed to parse event information from: {statement}")
                continue

    def _process_argument_question(self, qa: QAPair):
        """引数に関する質問を処理"""
        if (
            not qa.is_valid
            or qa.answer == "No arguments found."
            or not self.current_event
        ):
            return

        try:
            # "Argument is" で分割
            arg_statements = qa.answer.split("Argument is")
            arguments = []

            for statement in arg_statements[1:]:  # 最初の空の部分をスキップ
                try:
                    # テキストとロールを抽出
                    parts = statement.split("Role is")
                    if len(parts) != 2:
                        continue

                    text = parts[0].strip().strip(".")
                    role = parts[1].strip().strip(".")

                    if text and role:
                        arguments.append(
                            {
                                "text": text,
                                "role": role,
                            }
                        )

                except (IndexError, AttributeError):
                    print(f"Warning: Failed to parse argument from: {statement}")
                    continue

            if arguments:
                self.current_event["arguments"].extend(arguments)

        except Exception as e:
            print(f"Warning: Error processing arguments from answer: {qa.answer}")
            print(f"Error: {e!s}")

    def get_reconstructed_graph(self) -> dict[str, list[dict]]:
        """再構築されたグラフを取得"""
        # 空または不完全なイベントを除外
        valid_events = [
            event
            for event in self.events
            if event.get("trigger", {}).get("text") and event.get("event_type")
        ]
        return {"event_mentions": valid_events}


class GraphVisualizer:
    """グラフの可視化を行うクラス"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_graph(
        self, graph_data: dict[str, list[dict]], step: int, wnd_id: str
    ) -> None:
        """グラフを可視化してPNGとして保存"""
        G = nx.DiGraph()

        # ノードを追加
        for event in graph_data["event_mentions"]:
            # イベントノード
            event_id = f"Event_{event['trigger']['text']}"
            G.add_node(
                event_id,
                node_type="event",
                label=f"{event['trigger']['text']}\n({event['event_type']})",
            )

            # 引数ノード
            for arg in event["arguments"]:
                arg_id = f"Arg_{arg['text']}"
                if not G.has_node(arg_id):
                    G.add_node(arg_id, node_type="argument", label=f"{arg['text']}")
                G.add_edge(event_id, arg_id, label=arg["role"])

        if not G.nodes():  # グラフが空の場合
            return

        # 描画
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # イベントノードとエンティティノードを別々に描画
        event_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "event"]
        arg_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "argument"]

        # ノードの描画
        nx.draw_networkx_nodes(
            G, pos, nodelist=event_nodes, node_color="lightblue", node_size=3000
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=arg_nodes, node_color="lightgreen", node_size=2000
        )

        # エッジの描画
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

        # ラベルの描画
        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        # エッジラベルの描画
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        plt.title(f"Graph State at Step {step}")
        plt.axis("off")

        # 保存
        output_path = self.output_dir / f"{wnd_id}_step{step}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()


from tag_position import create_tagged_entity_mentions, create_tagged_event_mentions


def process_json_file(input_path: str, output_dir: str) -> None:
    """JSONファイルからイベントメンションを読み取り、グラフを生成して可視化する

    Args:
        input_path (str): 入力JSONファイルのパス
        output_dir (str): 出力ディレクトリのパス

    """
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSONファイルの読み込み
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # try:
            # 各行のJSONを解析
            data = json.loads(line.strip())

            if "event_mentions" not in data:
                print(f"Warning: No event_mentions found in line {line_num}")
                continue

            # tag_entity_mentions = tag_occurrecne_entity_mentions(data["sentence"], data["entity_mentions"])
            # tag_event_mentions = tag_occurrence_event_mentions(data["sentence"], data["event_mentions"], data["entity_mentions"])
            tag_entity_mentions = create_tagged_entity_mentions(
                data["sentence"], data["sentence_start"], data["entity_mentions"]
            )
            tag_event_mentions = create_tagged_event_mentions(
                data["sentence"],
                data["sentence_start"],
                data["event_mentions"],
                tag_entity_mentions,
            )

            # イベントグラフの作成（マージは自動的に行われる）
            event_graph = GENIAEventGraph(tag_event_mentions)

            # グラフの可視化
            fig = event_graph.visualize()
            wnd_id = data["wnd_id"]
            if wnd_id != "PMID-9032265-6":
                continue
            if fig is not None:  # フィギュアが生成された場合のみ保存
                wnd_id = data["wnd_id"]
                output_file = output_path / f"event_graph_{wnd_id}.png"
                fig.savefig(output_file, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Generated graph for {wnd_id}: {output_file}")
            else:
                print(
                    f"Warning: Could not generate graph for {data.get('wnd_id', 'unknown')}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON event mentions and create visualizations"
    )
    parser.add_argument("input_file", type=str, help="Input JSON file path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for visualizations (default: output)",
    )

    args = parser.parse_args()

    print(f"Processing file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    process_json_file(args.input_file, args.output_dir)
    print("Processing complete")


if __name__ == "__main__":
    main()
