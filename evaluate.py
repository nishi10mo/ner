# delete予定
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from flair.datasets import ColumnCorpus, DataLoader
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
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

def main():
    columns = {0: 'text', 1: 'ner'}
    data_folder = '.'
    corpus = ColumnCorpus(data_folder,
                        columns,
                        train_file='data/kwdlc/kwdlc_ner_train.conll',
                        dev_file='data/kwdlc/kwdlc_ner_validation.conll',
                        test_file='data/kwdlc/kwdlc_ner_test.conll')
    
    tagger = SequenceTagger.load('weights/final-model.pt')

    result, loss = tagger.evaluate(DataLoader(corpus.test, batch_size=1), gold_label_type="ner")
    output_scores(result.detailed_results, "visualize_results/scores/scores.json")

if __name__=="__main__":
    main()
