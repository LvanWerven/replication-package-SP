# This script was used to evaluate the predictions made by VisualNarrator against ground truth data.
# It calculates true positives, false negatives, false positives, precision, recall, and F-scores at different thresholds.

# In order to use it correctly, ensure that the predictions files are named and located according to the pattern "../../results/VisualNarrator/theshold{threshold}/{domain}/predictions_{domain}.txt"
# and that the ground truth data is available in the same folder as the script as 'ground_truth.csv'.

# Some parts of the script were created with the help of Gemini. Though all the logic was provided in the prompt used to generate the code, the code has been manually reviewed and adjusted as needed.

from typing import List
import pandas as pd
import os
from tabulate import tabulate

# === USER SETTINGS ===
DOMAINS = ["camperplus", "supermarket", "fish&chips", "planningpoker", "grocery", "school", "sports", "ticket"]  # update this list with your actual domains
THRESHOLDS = [1,2,3,4,5] # thresholds for VisualNarrator

base_predictions_VN_pattern = "../../results/VisualNarrator/{}/{}/predictions_{}.txt"    # naming pattern for predictions files

SUMMARIZE_RESULTS = True
BEST_SCORE_PER_DOMAIN = 'F1_Score'  # Metric to determine the best score per domain

CREATE_LATEX_TABLE = True 
OUTPUT_FILENAME = "main_table_results_VN.tex"  # Output LaTeX file name

TEST_SETTINGS = False  # Whether to run in test mode with limited domains

if TEST_SETTINGS:
    DOMAINS = ["camperplus"]
    THRESHOLDS = [1, 2]
    SUMMARIZE_RESULTS = False
    CREATE_LATEX_TABLE = False

ground_truth_all = pd.read_csv("../ground_truth.csv")  # columns: domain, Class, Singular, Type
print("Ground truth data loaded.", ground_truth_all.shape[0], "rows.")

synonyms_all = pd.read_csv("../synonyms.csv")  # columns: Class, Synonym
notpunish_all = pd.read_csv("../notpunish.csv")  # columns: domain, Class

def convert_to_latex_table(df: pd.DataFrame) -> None:
    """
    Converts a DataFrame to TWO SEPARATE LaTeX tables (Counts/P/R and F-Scores), 
    each in its own environment for flexible placement in the document.
    """
    
    df_copy = df.copy() 
    
    # ------------------------------------------------------------------
    # --- DYNAMIC STEP 1: Define ALL_HEADERS from the input DataFrame ---
    # ------------------------------------------------------------------
    
    ALL_HEADERS = df_copy.columns.to_list()
    
    try:
        r_s_index = ALL_HEADERS.index('Recall_Should_Have')
        f05_index = ALL_HEADERS.index('F0.5_Score')
        
    except ValueError as e:
        print(f"❌ Error: A key column is missing in your DataFrame index. Received error: {e}")
        print(f"Actual DataFrame Columns: {ALL_HEADERS}")
        return
    
    # Table A: Columns 0 (Domain) up to and including R.(S) (13 columns)
    HEADERS_A_NAMES = ALL_HEADERS[:r_s_index + 1]  
    # Table B: Columns 0-1 (Domain/Threshold) plus F-Scores from F0.5 onwards (11 columns)
    HEADERS_B_NAMES = ALL_HEADERS[:2] + ALL_HEADERS[f05_index:] 
    
    # ------------------------------------------------------------------
    # --- 1. PREP: Clean the DataFrame for tabulate (NO LaTeX Bolding) ---
    # ------------------------------------------------------------------
    
    mean_label = 'Mean' 
    df_copy.iloc[-1, 0] = mean_label 
    df_copy.iloc[-1, 1] = ''

    mean_values_all = []
    for col in ALL_HEADERS[2:]: 
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].round(3).astype(str)
        else:
            df_copy[col] = df_copy[col].astype(str)
        mean_values_all.append(df_copy.loc[df_copy.index[-1], col])
    
    # ------------------------------------------------------------------
    # --- 2. GENERATE: Create TWO DataFrames and LaTeX Body ---
    # ------------------------------------------------------------------
    
    # === Table A Generation ===
    df_A = df_copy.loc[:, HEADERS_A_NAMES]
    col_format_A = '|l|c|' + 'r|' * (len(df_A.columns) - 2)
    col_alignment_A = [c for c in col_format_A if c not in ['|', ' ']]
    latex_table_body_A = tabulate(df_A.values.tolist(), headers=HEADERS_A_NAMES, tablefmt="latex", showindex=False, colalign=col_alignment_A)
    
    # === Table B Generation ===
    df_B = df_copy.loc[:, HEADERS_B_NAMES]
    col_format_B = '|l|c|' + 'r|' * (len(df_B.columns) - 2)
    col_alignment_B = [c for c in col_format_B if c not in ['|', ' ']]
    latex_table_body_B = tabulate(df_B.values.tolist(), headers=HEADERS_B_NAMES, tablefmt="latex", showindex=False, colalign=col_alignment_B)

    # ------------------------------------------------------------------
    # --- 3. POST-PROCESS: Apply Bolding and Extract Content ---
    # ------------------------------------------------------------------
    
    # Apply Bolding
    for val in mean_values_all:
        if val in latex_table_body_A:
            latex_table_body_A = latex_table_body_A.replace(f' {val} ', f' \\textbf{{{val}}} ')
        if val in latex_table_body_B:
            latex_table_body_B = latex_table_body_B.replace(f' {val} ', f' \\textbf{{{val}}} ')

    latex_table_body_A = latex_table_body_A.replace(mean_label, f'\\textbf{{{mean_label}}}')
    latex_table_body_B = latex_table_body_B.replace(mean_label, f'\\textbf{{{mean_label}}}')
    
    # Extract only the content lines
    content_lines_A = latex_table_body_A.split('\n')[4:-1] 
    content_lines_B = latex_table_body_B.split('\n')[4:-1] 
    
    # ------------------------------------------------------------------
    # --- 4. ASSEMBLE: Two Independent Table Structures ---
    # ------------------------------------------------------------------
    
    # === Table A Structure (Counts, Precision, Recall) ===
    header_structure_A = f"""
\\hline
\\multicolumn{{2}}{{|c|}}{{\\raisebox{{-1.5ex}}[0pt][0pt]{{Domain $|$ Threshold}}}} & \\multicolumn{{5}}{{|c|}}{{\\textbf{{Counts}}}} & \\multicolumn{{3}}{{|c|}}{{\\textbf{{Precision}}}} & \\multicolumn{{3}}{{|c|}}{{\\textbf{{Recall}}}} \\\\
\\cline{{3-13}}
\\multicolumn{{2}}{{|c|}}{{}} & {ALL_HEADERS[2]} & {ALL_HEADERS[3]} & {ALL_HEADERS[4]} & {ALL_HEADERS[5]} & {ALL_HEADERS[6]} & {ALL_HEADERS[7]} & {ALL_HEADERS[8]} & {ALL_HEADERS[9]} & {ALL_HEADERS[10]} & {ALL_HEADERS[11]} & {ALL_HEADERS[12]} \\\\
\\hline
    """
    
    # We use \resizebox{\textwidth}{!} here to scale the wide table to fit the text width.
    # Note: If you still need a super-wide table, you can replace \textwidth with 1.25\textwidth 
    # and re-introduce the \makebox and minipage, but keep it self-contained.
    final_latex_table_A = f"""
\\begin{{table}}[h!]
\\noindent
\\caption{{Performance Metrics by Domain: Counts, Precision, and Recall}}
\\label{{tab:metrics_VN}}
\\fontsize{{6.5pt}}{{7.5pt}}\\selectfont 
\\makebox[\\linewidth][c]{{%
\\resizebox{{1.3\\textwidth}}{{!}}{{ 
    \\begin{{tabular}}{{{col_format_A}}}
    {header_structure_A}
    {'\\n'.join(content_lines_A)} 
    \\hline
    \\end{{tabular}}
}}
\\caption*{{(M): Must Have; (S): Should Have. P. and R. denote overall Precision and Recall. All values are shown rounded to 3 decimal places.}}
\\end{{table}}
"""

    # === Table B Structure (F-Scores) ===
    header_structure_B = f"""
\\hline
\\multicolumn{{2}}{{|c|}}{{\\raisebox{{-1.5ex}}[0pt][0pt]{{Domain $|$ Threshold}}}} & \\multicolumn{{3}}{{|c|}}{{\\textbf{{F$_{{0.5}}$ Score}}}} & \\multicolumn{{3}}{{|c|}}{{\\textbf{{F$_1$ Score}}}} & \\multicolumn{{3}}{{|c|}}{{\\textbf{{F$_2$ Score}}}} \\\\
\\cline{{3-11}}
\\multicolumn{{2}}{{|c|}}{{}} & F$_{{0.5}}$ & F$_{{0.5}}$(M) & F$_{{0.5}}$(S) & F$_1$ & F$_1$(M) & F$_1$(S) & F$_2$ & F$_2$(M) & F$_2$(S) \\\\
\\hline
    """
    
    final_latex_table_B = f"""
\\begin{{table}}[h!]
\\centering
\\caption{{Performance Metrics by Domain: F-Scores}}
\\label{{tab:metrics_VN_Fscores}}
\\fontsize{{6.5pt}}{{7.5pt}}\\selectfont 
\\resizebox{{\\textwidth}}{{!}}{{ 
    \\begin{{tabular}}{{{col_format_B}}}
    {header_structure_B}
    {'\\n'.join(content_lines_B)} 
    \\hline
    \\end{{tabular}}
}}}}
\\caption*{{(M): Must Have; (S): Should Have. All values are shown rounded to 3 decimal places.}}
\\end{{table}}
"""

    # --- Full LaTeX Environment Assembly ---
    # We wrap both tables in the same script for convenience.
    full_latex_script = f"""
\\documentclass[runningheads]{{llncs}}
\\usepackage{{booktabs}} 
\\usepackage{{amsmath}}
\\usepackage{{array}} 
\\usepackage{{graphicx}} 
\\usepackage{{adjustbox}} 
\\usepackage{{caption}} 
\\usepackage{{float}} 

\\begin{{document}}

\\section{{Results}} 

{final_latex_table_A}

{final_latex_table_B}

\\end{{document}}
"""

    # --- File Saving Block ---
    try:
        # Changed output filename to reflect that it contains multiple tables
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(full_latex_script)
        print(f"✅ Successfully saved two LaTeX tables to: {OUTPUT_FILENAME}.")
    except IOError as e:
        print(f"❌ Error writing to file {OUTPUT_FILENAME}: {e}")


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

def check_term_inclusion(df, search_term):
    """
    Checks if a given term is included in the 'concept' or any 'synonym' column
    of the DataFrame using vectorized pandas operations (case-insensitive, partial match).

    Args:
        df (pd.DataFrame): The DataFrame containing the concept and synonym columns.
        search_term (str): The term to search for.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows that match the search_term.
    """
    if not search_term:
        print("Please provide a search term.")
        return pd.DataFrame()

    # 2. Identify the columns to search
    search_cols = [col for col in df.columns if col == 'concept' or col.startswith('synonym')]

    # 3. Create a boolean mask using .stack() and .isin()
    # a. Select and stack: Convert the selected columns into a single long Series.
    # b. Normalize: Convert all values to lowercase.
    # c. Check membership: Use isin() for an exact match check.
    # d. Get the index: Since stack creates a MultiIndex (row_index, col_name), we need to check if
    #    the *row index* (level 0) is present in the matching items.
    matching_indices = df[search_cols].stack(future_stack=True).str.lower()
    
    mask_indices = matching_indices[matching_indices.isin([lower_term])].index.get_level_values(0).unique()
    
    # 4. Filter and return the matching rows
    matching_rows = df.loc[mask_indices].copy()

    if matching_rows['concept'].unique().size > 1:
        return Warning(f"Multiple matches found for term '{search_term}' in synonyms.")

    return matching_rows['concept'].unique()[0] if not matching_rows.empty else None

def fscore(beta, precision, recall):
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0

def get_ground_truth_of_concept(ground_truth_df, term):
    if term not in ground_truth['Class'].values and term not in ground_truth['Singular'].values:
        return None
    row = ground_truth_df[ground_truth_df['Class'] == term]
    if row.empty:
        row = ground_truth_df[ground_truth_df['Singular'] == term]
    return row.iloc[0]

best_score_per_domain = []

for domain in DOMAINS:
    print(f"\n=== Evaluating domain: {domain} ===")

    ground_truth = ground_truth_all[ground_truth_all['domain'] == domain]
    synonyms_domain = synonyms_all[synonyms_all['domain'] == domain]
    notpunish_domain = notpunish_all[notpunish_all['domain'] == domain]

    results_per_threshold = []

    for threshold in THRESHOLDS:
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
        used_synonym_concepts_set = set()

        for term in pred_list:
            lower_term = term.lower()

            synonym_concept = check_term_inclusion(synonyms_domain, lower_term)
            notpunish_concept = check_term_inclusion(notpunish_domain, lower_term)

            # A synonym was found, however the same concept has been used before
            if synonym_concept and synonym_concept in used_synonym_concepts_set:
                if notpunish_concept is not None:
                    annotated_rows.append({'Predicted': lower_term, 'Status': f'Not punished - a synonym for concept {synonym_concept}, which is in not punished list'})
                    continue
                fp += 1
                annotated_rows.append({'Predicted': lower_term, 'Status': f'FP - a synonym for concept {synonym_concept}, which is already used'})
                continue
            
            # A synonym was found, and the concept has not been used before
            elif synonym_concept:
                used_synonym_concepts_set.add(synonym_concept)
                ground_truth_row = get_ground_truth_of_concept(ground_truth, synonym_concept)
                if ground_truth_row is not None:
                    if ground_truth_row['Type'] == 'Must-have':
                      tp_must_have += 1
                      annotated_rows.append({'Predicted': lower_term, 'Status': f'TP - synonym for concept {synonym_concept}, which is Must-have'})
                    else:
                      tp_should_have += 1
                      annotated_rows.append({'Predicted': lower_term, 'Status': f'TP - synonym for concept {synonym_concept}, which is Should-have'})
                elif notpunish_concept is not None:
                    annotated_rows.append({'Predicted': lower_term, 'Status': f'Not punished - synonym for concept {synonym_concept}, which is in not punished list'})
                else:
                    fp += 1
                    annotated_rows.append({'Predicted': lower_term, 'Status': f'FP - synonym for concept {synonym_concept}, which is Not in ground truth'})
            
            # A synonym was not found, test for lower_term
            else:
                ground_truth_row = get_ground_truth_of_concept(ground_truth, lower_term)
                # The term is found directly in ground truth
                if ground_truth_row is not None:
                    if ground_truth_row['Type'] == 'Must-have':
                      tp_must_have += 1
                      annotated_rows.append({'Predicted': lower_term, 'Status': 'TP - concept is Must-have'})
                    else:
                      tp_should_have += 1
                      annotated_rows.append({'Predicted': lower_term, 'Status': 'TP - concept is Should-have'})
                    continue
                elif notpunish_concept is not None:
                    annotated_rows.append({'Predicted': lower_term, 'Status': f'Not punished - synonym for concept {synonym_concept}, which is in not punished list'})
                else:
                    fp += 1
                    annotated_rows.append({'Predicted': lower_term, 'Status': 'FP - Not in ground truth'})


        
        for _, row in ground_truth.iterrows():
            if row['Type'] == 'Must-have':
                if row['Class'] not in pred_list and row['Singular'] not in pred_list:
                    fn_must_have += 1
                    annotated_rows.append({'GroundTruth': row['Class'], 'Status': 'FN - concept is ground truth but not detected, type is must-have'})
            else:
                if row['Class'] not in pred_list and row['Singular'] not in pred_list:
                    fn_should_have += 1
                    annotated_rows.append({'GroundTruth': row['Class'], 'Status': 'FN - concept is ground truth but not detected, type is should-have'})

        filename = f"{domain}/annotated_concepts_threshold{threshold}_{domain}.txt"

        with open(filename, 'w') as f:
            for item in annotated_rows:
                f.write(str(item) + '\n')

        precision = (tp_must_have + tp_should_have) / (tp_must_have + tp_should_have + fp) if (tp_must_have + tp_should_have + fp) > 0 else 0
        recall = (tp_must_have + tp_should_have) / (tp_must_have + tp_should_have + fn_must_have + fn_should_have) if (tp_must_have + tp_should_have + fn_must_have + fn_should_have) > 0 else 0

        precision_must_have = tp_must_have / (tp_must_have + fp) if (tp_must_have + fp) > 0 else 0
        precision_should_have = tp_should_have / (tp_should_have + fp) if (tp_should_have + fp) > 0 else 0

        recall_must_have = tp_must_have / (tp_must_have + fn_must_have) if (tp_must_have + fn_must_have) > 0 else 0
        recall_should_have = tp_should_have / (tp_should_have + fn_should_have) if (tp_should_have + fn_should_have) > 0 else 0

        results_per_threshold.append({
            'Threshold': threshold,
            'TP_Must_Have': tp_must_have,
            'TP_Should_Have': tp_should_have,
            'FN_Must_Have': fn_must_have,
            'FN_Should_Have': fn_should_have,
            'FP': fp,
            'Precision': precision,
            'Precision_Must_Have': precision_must_have,
            'Precision_Should_Have': precision_should_have,
            'Recall': recall,
            'Recall_Must_Have': recall_must_have,
            'Recall_Should_Have': recall_should_have,
            'F0.5_Score': fscore(0.5, precision, recall),
            'F0.5_Score_Must_Have': fscore(0.5, precision_must_have, recall_must_have),
            'F0.5_Score_Should_Have': fscore(0.5, precision_should_have, recall_should_have),
            'F1_Score': fscore(1, precision, recall),
            'F1_Score_Must_Have': fscore(1, precision_must_have, recall_must_have),
            'F1_Score_Should_Have': fscore(1, precision_should_have, recall_should_have),
            'F2_Score': fscore(2, precision, recall),
            'F2_Score_Must_Have': fscore(2, precision_must_have, recall_must_have),
            'F2_Score_Should_Have': fscore(2, precision_should_have, recall_should_have),
        })

    highestThreshold = max(results_per_threshold, key=lambda x: x[BEST_SCORE_PER_DOMAIN])

    results_per_threshold.append({
        'Threshold': 'Best F1 Score at threshold ' + str(highestThreshold['Threshold'])
    })
    results = pd.DataFrame(results_per_threshold)
    results.to_csv(f"{domain}/evaluation_{domain}.csv", index=False)

    best_score_per_domain.append({
        'Domain': domain,
        **highestThreshold
    })


if SUMMARIZE_RESULTS:
    best_scores_df = pd.DataFrame(best_score_per_domain)

    precision_mean = best_scores_df['Precision'].mean()
    precision_must_have = best_scores_df['Precision_Must_Have'].mean()
    precision_should_have = best_scores_df['Precision_Should_Have'].mean()
    recall_mean = best_scores_df['Recall'].mean()
    recall_must_have = best_scores_df['Recall_Must_Have'].mean()
    recall_should_have = best_scores_df['Recall_Should_Have'].mean()


    summary = {
        "Domain": "Mean of all the domains best runs based on " + BEST_SCORE_PER_DOMAIN,
        'TP_Must_Have': best_scores_df['TP_Must_Have'].mean(),
        'TP_Should_Have': best_scores_df['TP_Should_Have'].mean(),
        'FN_Must_Have': best_scores_df['FN_Must_Have'].mean(),
        'FN_Should_Have': best_scores_df['FN_Should_Have'].mean(),
        'FP': best_scores_df['FP'].mean(),
        'Precision': precision_mean,
        'Precision_Must_Have': precision_must_have,
        'Precision_Should_Have': precision_should_have,
        'Recall': recall_mean,
        'Recall_Must_Have': recall_must_have,
        'Recall_Should_Have': recall_should_have,
        'F0.5_Score': fscore(0.5, precision_mean, recall_mean),
        'F0.5_Score_Must_Have': fscore(0.5, precision_must_have, recall_must_have),
        'F0.5_Score_Should_Have': fscore(0.5, precision_should_have, recall_should_have),
        'F1_Score': fscore(1, precision, recall),
        'F1_Score_Must_Have': fscore(1, precision_must_have, recall_must_have),
        'F1_Score_Should_Have': fscore(1, precision_should_have, recall_should_have),
        'F2_Score': fscore(2, precision, recall),
        'F2_Score_Must_Have': fscore(2, precision_must_have, recall_must_have),
        'F2_Score_Should_Have': fscore(2, precision_should_have, recall_should_have),
    }

    summary_df = pd.DataFrame([summary])

    best_scores_df = pd.concat([best_scores_df, summary_df], ignore_index=True)
    best_scores_df.to_csv("best_scores_per_domain.csv", index=False)

if CREATE_LATEX_TABLE:
    convert_to_latex_table(best_scores_df)

        

        

