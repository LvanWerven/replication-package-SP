# This script was used to evaluate the predictions made by VisualNarrator against ground truth data.
# It calculates true positives, false negatives, false positives, precision, recall, and F-scores at different thresholds.

# In order to use it correctly, ensure that the predictions files are named according to the pattern "theshold{threshold}/{domain}/predictions_{domain}.txt"
# and that the ground truth data is available in the same folder as 'ground_truth.csv'.


from typing import List
import pandas as pd
import os

# === USER SETTINGS ===
domains = ["camperplus", "supermarket", "fish&chips", "planningpoker", "grocery", "school", "sports", "ticket"]  # update this list with your actual domains
base_predictions_VN_pattern = "{}/{}/predictions_{}.txt"    # naming pattern for predictions files
thresholds = [1,2,5] # thresholds for VisualNarrator

ground_truth_all = pd.read_csv("ground_truth.csv")  # columns: domain, Class, Singular, Type

def txt_file_to_list(file_path: str) -> List[str]:
    """
    Reads a text file and returns a list of strings, where each string
    corresponds to a line in the file. Newline characters are stripped.

    Args:
        file_path (str): The full path to the .txt file.

    Returns:
        List[str]: A list of strings, or an empty list if the file is not found.
    """

    absolute_path = os.path.abspath(file_path)

    # 1. Check if the file exists
    if not os.path.exists(absolute_path):
        print(f"❌ Error: File not found at path: {absolute_path}")
        return []

    data_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                
                if not cleaned_line or cleaned_line.startswith("Total"):
                    continue # Skip this line and move to the next iteration
                
                data_list.append(cleaned_line.lower().strip())
        return data_list

    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return []

def fscore(beta, precision, recall):
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0


for domain in domains:
    print(f"\n=== Evaluating domain: {domain} ===")

    ground_truth = ground_truth_all[ground_truth_all['domain'] == domain]

    results_per_threshold = []

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")

        threshold_str = f"threshold{threshold}"

        pred_file = base_predictions_VN_pattern.format(threshold_str, domain, domain)
        print(f"Loading predictions from: {pred_file}")
        pred_list = txt_file_to_list(pred_file)
        if not pred_list:
            print(f"⚠️ Predictions file not found for domain '{domain}' — skipping.")
            continue

        tp_must_have = 0
        tp_should_have = 0
        fn_must_have = 0
        fn_should_have = 0
        fp = 0

        annotated_rows = []

        for term in pred_list:
            lower_term = term.lower()
            if lower_term not in ground_truth['Class'].values and lower_term not in ground_truth['Singular'].values:
                fp += 1
                annotated_rows.append({'Predicted': term, 'Status': 'FP - Not in ground truth'})
            elif lower_term in ground_truth['Class'].values or lower_term in ground_truth['Singular'].values:
                row = ground_truth[ground_truth['Class'] == lower_term]
                if row.empty:
                    row = ground_truth[ground_truth['Singular'] == lower_term]
                row = row.iloc[0]
                if row['Type'] == 'Must-have':
                    tp_must_have += 1
                    annotated_rows.append({'Predicted': lower_term, 'Status': 'TP - concept is Must-have'})
                else:
                    tp_should_have += 1
                    annotated_rows.append({'Predicted': lower_term, 'Status': 'TP - concept is Should-have'})
            else : 
                print(f"term '{term}' not found.")
        
        for _, row in ground_truth.iterrows():
            if row['Type'] == 'Must-have':
                if row['Class'] not in pred_list and row['Singular'] not in pred_list:
                    fn_must_have += 1
                    annotated_rows.append({'GroundTruth': row['Class'], 'Status': 'FN - concept is ground truth but not detected, type is must-have'})
            else:
                if row['Class'] not in pred_list and row['Singular'] not in pred_list:
                    fn_should_have += 1
                    annotated_rows.append({'GroundTruth': row['Class'], 'Status': 'FN - concept is ground truth but not detected, type is should-have'})

        filename = f"annotated_concepts_threshold{threshold}_{domain}.txt"

        with open(filename, 'w') as f:
            for item in annotated_rows:
                f.write(str(item) + '\n')

        precision = (tp_must_have + tp_should_have) / (tp_must_have + tp_should_have + fp) if (tp_must_have + tp_should_have + fp) > 0 else 0
        recall = (tp_must_have + tp_should_have) / (tp_must_have + tp_should_have + fn_must_have + fn_should_have) if (tp_must_have + tp_should_have + fn_must_have + fn_should_have) > 0 else 0

        results_per_threshold.append({
            'Threshold': threshold,
            'TP_Must_Have': tp_must_have,
            'TP_Should_Have': tp_should_have,
            'FN_Must_Have': fn_must_have,
            'FN_Should_Have': fn_should_have,
            'FP': fp,
            'Precision': precision,
            'Recall': recall,
            'F0.5_Score': fscore(0.5, precision, recall),
            'F1_Score': fscore(1, precision, recall),
            'F2_Score': fscore(2, precision, recall)
        })

    results = pd.DataFrame(results_per_threshold)
    results.to_csv(f"evaluation_{domain}.csv", index=False)




        

        

