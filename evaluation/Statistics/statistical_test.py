# This script performs the Wilcoxon Signed-Rank Test to compare the performance of the two techniques
# The script was created using Gemini, it has been evaluated by the authors to make sure it meets the requirements.

import pandas as pd
from scipy.stats import wilcoxon

SIGNIFICANT_THRESHOLD = 0.05
OUTPUT_FILENAME = "wilcoxon_test_results.txt"

mapping_of_names = pd.read_csv("./mapping_results_LLM_VN.csv")

results_LLM = pd.read_csv("../LLM/script/overall_domain_summary.csv")
results_VN = pd.read_csv("../VisualNarrator/best_scores_per_domain.csv")

results_LLM = results_LLM.set_index("Domain").sort_index()
results_VN = results_VN.set_index("Domain").sort_index()

def output(*args, **kwargs):
    """
    Custom output function to output messages.
    """
    print(*args, **kwargs)
    # Write to file if the file object is open
    if 'file_out' in globals() and globals()['file_out']:
        # Capture the output to a string
        output = ' '.join(map(str, args))
        
        # Add newline character if one wasn't suppressed
        if 'end' not in kwargs or kwargs['end'] == '\n':
            output += '\n'
            
        globals()['file_out'].write(output)


def run_wilcoxon_test(data1, data2, metric_name, rq_description):
    """
    Performs the Wilcoxon Signed-Rank Test and outputs the result.
    """

    # Get the corresponding LLM metric name
    metric_name_LLM = mapping_of_names[mapping_of_names['name'] == metric_name]['LLM_name'].values[0]
    # metric_name_LLM = metric_name_mapping['LLM_name'].values[0]

    # Align the dataframes by the common index (Domain)
    df_combined = pd.DataFrame({
        'LLM_Score': data1[metric_name_LLM].astype(float),
        'VN_Score': data2[metric_name].astype(float)
    })
    
    # Drop any rows where either score is missing (though should not happen here)
    df_combined = df_combined.dropna()
    
    # Calculate the test statistic (W) and p-value
    statistic, p_value = wilcoxon(df_combined['LLM_Score'], df_combined['VN_Score'])

    # Determine conclusion
    if p_value < SIGNIFICANT_THRESHOLD:
        conclusion = f"REJECT H0: There IS a statistically significant difference (p < {SIGNIFICANT_THRESHOLD})."
    else:
        conclusion = f"FAIL TO REJECT H0: There is NO statistically significant difference (p >= {SIGNIFICANT_THRESHOLD})."
        
    output(f"\n--- {rq_description} ---")
    output(f"H0: No significant difference between techniques for {metric_name}.")
    output(f"Test Metric: F1-Score (Technique A vs Technique B)")
    output(f"Wilcoxon W Statistic: {statistic:.4f}")
    output(f"P-value (Exact): {p_value:.4f}")
    output(f"Conclusion: {conclusion}")
    output("--------------------------------------------------")
    
    return statistic, p_value

with open(OUTPUT_FILENAME, 'w') as file_out:
    
    output(f"\n*** Applying Wilcoxon Signed-Rank Test (Alpha = {SIGNIFICANT_THRESHOLD}) ***")
    output("Data used: Paired domain scores.")

    # --------------------------------------------------
    # RQMain: Overall Performance
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F1_Score', 
        rq_description='Main RQ: Overall Performance'
    )

    # --------------------------------------------------
    # RQ1: Must-Have Classes Performance
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F1_Score_Must_Have', 
        rq_description='RQ1: Must-Have Classes Performance'
    )

    # --------------------------------------------------
    # RQ2: Should-Have Classes Performance
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F1_Score_Should_Have', 
        rq_description='RQ2: Should-Have Classes Performance'
    )


    # --------------------------------------------------
    # RQMain: Overall Performance F0.5
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F0.5_Score', 
        rq_description='Main RQ: Overall Performance F0.5'
    )

    # --------------------------------------------------
    # RQ1: Must-Have Classes Performance F0.5
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F0.5_Score_Must_Have', 
        rq_description='RQ1: Must-Have Classes Performance F0.5'
    )

    # --------------------------------------------------
    # RQ2: Should-Have Classes Performance F0.5
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F0.5_Score_Should_Have', 
        rq_description='RQ2: Should-Have Classes Performance F0.5'
    )
    
    # --------------------------------------------------
    # RQMain: Overall Performance F2
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F0.5_Score', 
        rq_description='Main RQ: Overall Performance F2'
    )

    # --------------------------------------------------
    # RQ1: Must-Have Classes Performance F2
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F0.5_Score_Must_Have', 
        rq_description='RQ1: Must-Have Classes Performance F2'
    )

    # --------------------------------------------------
    # RQ2: Should-Have Classes Performance F2
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='F0.5_Score_Should_Have', 
        rq_description='RQ2: Should-Have Classes Performance F2'
    )

    # --------------------------------------------------
    # RQMain: Overall Performance Precision
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='Precision', 
        rq_description='Main RQ: Overall Performance Precision'
    )

    # --------------------------------------------------
    # RQ1: Must-Have Classes Performance Precision
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='Precision_Must_Have', 
        rq_description='RQ1: Must-Have Classes Performance Precision'
    )

    # --------------------------------------------------
    # RQ2: Should-Have Classes Performance Precision
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='Precision_Should_Have', 
        rq_description='RQ2: Should-Have Classes Performance Precision'
    )

    # --------------------------------------------------
    # RQMain: Overall Performance Recall
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='Recall', 
        rq_description='Main RQ: Overall Performance Recall'
    )

    # --------------------------------------------------
    # RQ1: Must-Have Classes Performance Recall
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='Recall_Must_Have', 
        rq_description='RQ1: Must-Have Classes Performance Recall'
    )

    # --------------------------------------------------
    # RQ2: Should-Have Classes Performance Recall
    # --------------------------------------------------
    run_wilcoxon_test(
        results_LLM, 
        results_VN, 
        metric_name='Recall_Should_Have', 
        rq_description='RQ2: Should-Have Classes Performance Recall'
    )

print(f"\n✅ Script finished. All output has been saved to: {OUTPUT_FILENAME}")
