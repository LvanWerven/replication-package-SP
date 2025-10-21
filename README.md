# Replication-package-SP
The replication package used for our research paper in the course Software Production 2025


# Content
* data
    * labeling_classes.csv : the excel file that enabled the creation of the ground truth concept classification into the must_have and should_have labels.
* evaluation
    * LLM
        * script: containing the annotation files for each domain and each execution round, the evaluation script and the average per domain and overall average
        * evaluation csvs:  for each domain containing the metrics for each round of execution for the specific domain
    * Statistics
        * mapping_results_LLM_VN.csv : a file mapping the column names of the summary csv's of each technique. Used in the statistical_test.py script
        * statistical_test.py: The script applying the wilcoxon signed-rank test on the f1-score's of both techniques
        * wilcoxon_test_results.txt: the file containing the output of the statistical test script
    * VisualNarrator:
        * for each domain: 
            * the annotated concepts used to calculate the scores per threshold
            * evaluation_domain.csv: the evaluation metrics per threshold 
        * best_scores_per_domain.csv: a summary of per domain the best scoring threshold, based on f1-score and the scores with the average combining all the metrics per domain
        * evaluate.py: The script creating the evaluation from the results of the VN and creating a two latex tables output the results
        * main_table_results_VN.tex: The latex tables holding the VN results
* paper-artifacts
    * research-problem-extended.md: consisting of the sub research questions with hypothesis
* Prompt LLM
    * The prompts used in the extraction of LLM results
* results
    * VisualNarrator
        * threshold x: three folders with the results of Visual Narrator runs on the input with different thresholds
        * HowToRun.md: An detailed explanation of how we run the visualnarrator to get the results. 
        * rename.py: a script to rename the output files of the Visual Narrator based on the folder that they are stored in and their extension.
