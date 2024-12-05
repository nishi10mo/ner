import os
import re
import json
import flair
import torch
import pandas as pd
import matplotlib.pyplot as plt
from flair.datasets import ColumnCorpus, DataLoader
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.data import Sentence, Dictionary
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import pdb

def output_scores(result, output_json_file):
    data = {}
    
    # MICRO_AVG と MACRO_AVG をパース
    micro_match = re.search(r'MICRO_AVG: acc ([\d.]+) - f1-score ([\d.]+)', result)
    macro_match = re.search(r'MACRO_AVG: acc ([\d.]+) - f1-score ([\d.]+)', result)
    if micro_match:
        data["MICRO_AVG"] = {
            "acc": float(micro_match.group(1)),
            "f1-score": float(micro_match.group(2))
        }
    if macro_match:
        data["MACRO_AVG"] = {
            "acc": float(macro_match.group(1)),
            "f1-score": float(macro_match.group(2))
        }

    # タグごとのスコアをパース
    tag_matches = re.findall(
        r'(\S+)\s+tp: (\d+) - fp: (\d+) - fn: (\d+) - tn: (\d+) - precision: ([\d.]+) - recall: ([\d.]+) - accuracy: ([\d.]+) - f1-score: ([\d.]+)',
        result
    )
    tags = {}
    for match in tag_matches:
        tag = match[0]
        tags[tag] = {
            "tp": int(match[1]),
            "fp": int(match[2]),
            "fn": int(match[3]),
            "tn": int(match[4]),
            "precision": float(match[5]),
            "recall": float(match[6]),
            "accuracy": float(match[7]),
            "f1-score": float(match[8])
        }

    data["tags"] = tags

    with open(output_json_file, "w") as f:
        json.dump(data, f)

def show_results(log_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

    # ログファイルを読み込む
    data = pd.read_csv(log_file, sep='\t')

    # エポックごとに損失とF1スコアをプロット
    plt.figure(figsize=(12, 6))

    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(data['EPOCH'], data['TRAIN_LOSS'], label='Training Loss', marker='o')
    plt.plot(data['EPOCH'], data['DEV_LOSS'], label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # F1スコア曲線
    plt.subplot(1, 2, 2)
    plt.plot(data['EPOCH'], data['DEV_F1'], label='Validation F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()

    # レイアウト調整
    plt.tight_layout()

    # 画像ファイルとして保存
    output_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(output_path, dpi=300)
    plt.close()  # プロットを閉じる

def main():

    # 各カラムに対応するデータの属性を定義（0列目がテキスト、1列目が固有表現ラベル）
    # columns = {0: 'id', 1: 'text', 2: 'ner', 3: 'other'}
    columns = {0: "text", 1: "ner", 2: "other"}

    # データセットが格納されているフォルダを指定
    data_folder = 'data/kwdlc/'

    # コーパス（データセット）を読み込み、学習、検証、テストデータを準備
    corpus = ColumnCorpus(data_folder,
                        columns,
                        train_file='kwdlc_ner_train.tsv',
                        dev_file='kwdlc_ner_validation.tsv',
                        test_file='kwdlc_ner_test.tsv')

    # 固有表現（NER）のラベルタイプを指定
    tag_type = 'ner'

    # コーパスからラベル辞書を作成（ラベルごとのIDを管理する）
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type, add_unk=True)

    # # 使用する埋め込み（特徴量）を定義
    # embedding_types = [
    #     FlairEmbeddings('ja-forward'),
    #     FlairEmbeddings('ja-backward'),
    # ]
    # # 埋め込みをスタック（組み合わせて利用）
    # embeddings = StackedEmbeddings(embeddings=embedding_types)

    # embeddings = TransformerWordEmbeddings(model='tohoku-nlp/bert-base-japanese-v3',
    embeddings = TransformerWordEmbeddings(model='FacebookAI/xlm-roberta-base',
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

    # 固有表現タグ付けモデルを作成
    tagger = SequenceTagger(hidden_size=256,  # 隠れ層のサイズ
                            embeddings=embeddings,  # 使用する埋め込み
                            tag_dictionary=tag_dictionary,  # ラベル辞書
                            tag_type=tag_type,  # タグの種類（NER）
                            use_crf=False, # CRF（条件付きランダムフィールド）を使用しない
                            use_rnn=False, # RNNを使用しない
                            tag_format="BIO",
                            reproject_embeddings=False,
                            allow_unk_predictions=True
                            )

    # トレーナーを作成し、モデルを学習
    trainer = ModelTrainer(tagger, corpus)
    trainer.fine_tune('weights/',  # 学習済みモデルの重みを保存するディレクトリ
                monitor_test=False,  # テストデータでの評価をスキップ
                learning_rate=0.1,  # 学習率
                mini_batch_size=32,  # ミニバッチのサイズ
                max_epochs=2)  # エポック数（学習回数）

    # 学習中の損失を可視化（`loss.tsv` を読み込み、結果を保存）
    show_results('weights/loss.tsv', 'visualize_results/loss')

    # テストデータを使ってモデルを評価
    # result, loss = tagger.evaluate(DataLoader(corpus.test, batch_size=1))
    result, loss = tagger.predict(DataLoader(corpus.test, batch_size=1))

    # 詳細な評価結果をJSONファイルに保存
    output_scores(result.detailed_results, "visualize_results/scores/scores.json")


if __name__=="__main__":
    main()
