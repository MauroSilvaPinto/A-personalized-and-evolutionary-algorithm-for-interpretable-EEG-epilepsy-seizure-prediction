## A personalized and evolutionary algorithm for interpretable EEG epilepsy seizure prediction

This is the code used for the paper "OA personalized and evolutionary algorithm for interpretable EEG epilepsy seizure prediction". It is a patient-specific Evolutionary Algorithm  for predicting epileptic seizures with the EEG signal, from data preprocessing to phenotype study.

## Code Organization Folders

- Data Processing
- Evolutionary Algorithm
- Phenotype Analysis

## Data Folders

- Processed_data
- Evolutionary_executions

## Preprocessing

You can not execute this code as it is necessary the raw data from EEG recordings. As the used dataset belongs to EPILEPSIAE, we can not make it publicly available online due to ethical concerns. We can only offer the extracted first-level features from non-overlapping windows of 5 seconds. In preprocessing code:
- [preprocessing.py] - the chunk of matlab code to extract the first-level features, in matlab.

## Evolutionary Algorithm

You can execute all the following scripts on patient 53402 with the preprocessed files we present. You can also skip the execution and check the 30 performed runs from the paper, which are present in Evolutionary_executions folder. Here are the scripts you can run:

- [main.py]: to execute the EA.
- [get_testing_results.py]: to get the EA results in new tested seizures.
- [get_training_results.py]: to get information on the selected individuals of the executed EA.
- [get_testing_results_and_surrogate_analysis.py]: besides testing results, it also presents the surrogate analysis.

# Phenotype Analysis

You can execute all the following scripts on patient 53402 with the data from Evolutionary_executions folders. These scripts perform the phenotype study, whose outputs are the graphs from the paper. Here are the scripts you need to run:

- [phenotype_study.py]: this script provides the figures from individual gene presence and gene power, which are provided in th paper and Supplementary Material.
 Attention: to run the phenotype_study.py script, you need to have it in the Evolutionary Algorithm folder, along with Analyze.py file.


## License
This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

You are free to use any of this material.

Just cite it as:
Pinto, M.F., Leal, A., Lopes, F. et al. A personalized and evolutionary algorithm for interpretable EEG epilepsy seizure prediction. Sci Rep 11, 3415 (2021). https://doi.org/10.1038/s41598-021-82828-7

