import os
import json
import matplotlib.pyplot as plt

def diff_train_data_num():
    # ディレクトリとファイルパターン
    scores_dir = "./visualize_results/scores"
    files = [f for f in os.listdir(scores_dir) if f.startswith("scores_") and f.endswith(".json")]

    # 語彙数とf1_scoreを格納するリスト
    vocab_sizes = []
    f1_scores = []

    # ファイルを読み込み、必要な情報を抽出
    for file in files:
        filepath = os.path.join(scores_dir, file)
        with open(filepath, 'r') as f:
            data = json.load(f)
            vocab_size = int(file.split("_")[1].split(".")[0])  # ファイル名から語彙数を抽出
            f1_score = data["MICRO_AVG"]["f1-score"]  # MICRO_AVGのf1_scoreを抽出
            vocab_sizes.append(vocab_size)
            f1_scores.append(f1_score)

    # 語彙数の昇順にソート
    sorted_data = sorted(zip(vocab_sizes, f1_scores))
    vocab_sizes, f1_scores = zip(*sorted_data)

    # グラフを作成
    plt.figure(figsize=(8, 6))
    plt.plot(vocab_sizes, f1_scores, marker="o")
    plt.title("F1 Score vs Vocabulary Size", fontsize=14)
    plt.xlabel("Vocabulary Size", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("visualize_results/f1score_graph/f1score_vs_traindata.png")

def diff_label_num():
    # ファイルのパス
    frequence_file = "./visualize_results/count/frequence.json"
    scores_file = "./visualize_results/scores/scores_202313.json"
    output_image_path = "./visualize_results/f1score_graph/f1score_vs_frequency.png"

    # データの読み込み
    with open(frequence_file, "r") as f:
        frequence_data = json.load(f)

    with open(scores_file, "r") as f:
        scores_data = json.load(f)

    # B-から始まるラベルの出現頻度を抽出
    b_frequencies = {key: value for key, value in frequence_data.items() if key.startswith("B-")}

    # B-を除いたラベル名に対応するf1スコアを抽出
    labels = [key[2:] for key in b_frequencies.keys()]  # ラベル名から "B-" を除去
    frequencies = list(b_frequencies.values())

    f1_scores = []
    for label in labels:
        if label in scores_data["tags"]:
            f1_scores.append(scores_data["tags"][label]["f1-score"])
        else:
            f1_scores.append(0.0)  # f1スコアが存在しない場合は0

    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.scatter(frequencies, f1_scores, color="blue", label="Labels")
    for i, label in enumerate(labels):
        plt.text(frequencies[i], f1_scores[i], label, fontsize=9, ha="right")

    plt.title("F1 Score vs Label Frequency", fontsize=14)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 1)  # 縦軸を0から1に設定
    plt.grid(True)
    plt.tight_layout()

    # 画像として保存
    plt.savefig(output_image_path)

def main():
    diff_train_data_num()
    diff_label_num()

if __name__=="__main__":
    main()
