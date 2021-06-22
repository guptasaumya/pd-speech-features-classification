# pd-speech-features-classification
Parkinson’s disease classification using speech signal features; comparison of various multiclass classification algorithms.

## Description

Parkinson’s disease (PD) causes motor system disorders, for example – speech disorder. According to Okan et al. (2019), 90% of the times, vocal impairments are exhibited in the early stages of the disease. Hence, several studies process speech signals, extract features valuable for PD assessments and build reliable decision support systems. This analysis is related to performing PD classification for humans using extracted features from speech signals of diseased and non-diseased humans. In this analysis, I do not focus on feature extraction methods. The task is classification using these features and comparative analysis of algorithm performance for the two cases mentioned below. Hence, the report does not explain the details or the concepts related to feature extraction or the features themselves. I have used the dataset developed in the analysis from Okan et al. (2019) to accomplish two tasks:

1. Perform and compare classification results from five algorithms – Decision Tree, Support-Vector Machine, Bagging approach, Random Forest, and Boosting method.

3. Use Principal Component Analysis (PCA) and perform the same classification task. Discuss the difference in the performance of algorithms between using the original features versus using the principal components.

This analysis was done as part of the ```Statistical Learning``` course held at Dalarna University for the master in the data science program. 

## Getting Started

### Usage

The analysis is presented in ```PDSpeechFeaturesClassificationReport.pdf``` and the classification code with algorithms in ```classification_model.R```.  The code is in R; hence, R and optional tools like RStudio are required to run the code. In addition, the images used in the report are present separately for easy access to the information.

However, this repository does not provide the data used for analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
