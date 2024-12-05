
import json
from collections import Counter

def count_label_frequencies(conll_file, output_json):
    """
    Count the frequencies of each label in a CoNLL file and save them in JSON format.

    Args:
        conll_file (str): Path to the input CoNLL file.
        output_json (str): Path to save the output JSON file.
    """
    label_counter = Counter()
    
    # Read the CoNLL file line by line
    with open(conll_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:  # Non-empty lines contain tokens and labels
                parts = line.split()
                if len(parts) == 2:  # Ensure the line has at least a token and label
                    token, label = parts
                    # Only count labels that are not "O"
                    if label != "O":
                        label_counter[label] += 1
    
    # Convert to JSON format
    label_frequencies = dict(label_counter)
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(label_frequencies, json_file, ensure_ascii=False, indent=4)
    
    print(f"Label frequencies saved to {output_json}")

# Example usage
input_conll = 'data/kwdlc/kwdlc_ner_train.conll'  # Input CoNLL file path
output_json = "visualize_results/count/frequence.json"  # Output JSON file path
count_label_frequencies(input_conll, output_json)
