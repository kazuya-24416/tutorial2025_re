# 関係抽出（Relation Extraction）チュートリアル

関係抽出（Relation Extraction）タスクのチュートリアル用に設計されています。自然言語処理における関係抽出の基本から実装までを学ぶことができます。

## プロジェクト構造

```
.
├── config
│   └── config.yaml       # 設定ファイル
├── data                  # データセットを格納するディレクトリ
├── docker
│   ├── Dockerfile        # Dockerコンテナ構築ファイル
│   ├── requirements.txt  # 必要なPythonパッケージリスト
├── src
│   ├── __init__.py
│   ├── compute_metrics_re.py    # 評価指標の計算
│   ├── custom_collator.py       # データの前処理用カスタムコレータ
│   ├── preprocess
│   │   ├── __init__.py
│   │   └── create_sft_data.py   # 学習データの作成スクリプト
│   ├── train_seq2seq.py         # Seq2Seqモデルの学習スクリプト
│   └── utils.py                 # ユーティリティ関数
└── tests                        # テストコード

```

## 概要

このチュートリアルでは、テキスト内のエンティティ間の関係を抽出する関係抽出（Relation Extraction）タスクに取り組みます。例えば「スティーブ・ジョブズはAppleの共同創業者です」というテキストから、[スティーブ・ジョブズ, 共同創業者, Apple]という関係を抽出することを学びます。
