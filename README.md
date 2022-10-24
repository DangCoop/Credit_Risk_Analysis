# Credit Risk Analysis 
____
![](/Images/Front1.jpeg)

***Utilized several machine learning models to predict credit risk using Python's imbalanced-learn and scikit-learn libraries***

## Supervised Machine Learning: Overview of the analysis
____
The goal of this project was to use machine learning models for credit risk analysis in order to provide faster and more reliable credit. As a basis, we took a set of credit card data from LendingClub, a peer-to-peer lending company. I was able to identify good loan candidates who could help the financial institution reduce its default rate on its loans. I was able to create and evaluate several machine learning algorithms for predicting credit risk.

The dataset consists of approximately 85 features (also known as variables) for each loan. Some examples are ***Principal and Interest Received to Date, Last Payment Amount, Interest Rate, Debt-to-Income Ratio, Number of Months Since the Last Loan Request, and Home Ownership.***

The problem inherent in this data set is that there is a serious imbalance in favor of good loans. Naturally, most loans will never default, resulting in 99.9% of the database being low risk loans. This is extremely incorrect information.

To overcome data skewness, we use sklearn to split the data into training and test sets. We then use the testing data to train various models and make predictions.

We evaluate the performance of the model by the following parameters:

 - [x] The accuracy score is simply the percentage of correct predictions, with 1 representing 100% accuracy and 0 representing 0% accuracy.
- [x] Precision Score - a measure of how reliable a positive classification is, with 1 being 100% reliable and 0 being 0% reliable. For example, "I know that the test for high risk came back positive. How likely is it that the loan is high risk?"
- [x] Recall Score - a measure of how many actual positives were identified correctly, with 1 being 100% correct and 0 being 0% correct. "I know that my loan is high risk. How likely is it that the test will predict it?"

## Resources
___
 - Dataset:
    - LoanStats_2019Q1.csv
- Software:  
    - Python
    - VS Code
    - NumPy, scikit-learn, and imbalanced-learn libraries
    - Logistic Regression and Random Forest models
    - Ensemble and resampling techinques
  
## Results
____
The results for all six machine learning algorithms are shown below with their outputs supported with images.

## Random Oversampling

- In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.
- Accuracy: 0.61 of high risk applications were predicted and actually correct.
![](/Images/Random%20Oversampling%20Balanced%20Acc_Score.png)
- Precision: 0.01 of high risk applications were predicted and actually correct.
- Recall: 0.63 of actual high risk applications identified correctly.
![](/Images/Random%20Oversampling%20Imbalanced%20Class_report.png)

## SMOTE Oversampling

- In synthetic minority oversampling technique (SMOTE), the size of the minority is increased by new instances being interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen.
- Accuracy: 0.61 of high risk applications were predicted and actually correct.
![](/Images/SMOTE%20Oversampling%20Balanced%20Acc_Score.png)

- Precision: 0.01 of high risk applications were predicted and actually correct.
- Recall: 0.55 of actual high risk applications identified correctly.
![](/Images/SMOTE%20Oversampling%20Imbalanced%20Class_report.png)

## Cluster Centroids Undersampling
- Cluster Centroilds identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.
- Accuracy: 0.65 of high risk applications were predicted and actually correct.
  ![](/Images/Cluster%20Centroids%20Undersampling%20Balanced%20Acc_Score.png)
- Precision: 0.01 of high risk applications were predicted and actually correct.
- Recall: 0.63 of actual high risk applications identified correctly.
![](/Images/Cluster%20Centroids%20Undersampling%20Imbalanced%20Class_report.png)

## SMOTEENN Combination Sampling
- SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. 
SMOTEENN is a two-step process: 
   1. Oversample the minority class with SMOTE.
   2. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
- Accuracy: 0.64 of high risk applications were predicted and actually correct.
![](/Images/SMOTEENN%20Combination%20Sampling%20Imbalanced%20Acc_Score.png)
- Precision: 0.01 of high risk applications were predicted and actually correct.
- Recall: 0.70 of actual high risk applications identified correctly.
![](/Images/SMOTEENN%20Combination%20Sampling%20Imbalanced%20Class_report.png)
## Random Forest Classifier

- The random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.
- Accuracy: 0.87 of high risk applications were predicted and actually correct.
![](/Images/Balanced%20Random%20Forest%20Acc_Score.png)

- Precision: 0.03 of high risk applications were predicted and actually correct.
- Recall: 0.70 of actual high risk applications identified correctly.
![](/Images/Balanced%20Random%20Forest%20Class_report.png)

## Easy Ensemble Classifier

- The Easy Ensemble involves creating balanced samples of the training dataset by selecting all examples from the minority class and a subset from the majority class. Rather than using pruned decision trees, boosted decision trees are used on each subset, specifically the AdaBoost algorithm.
- Accuracy: 0.94 of high risk applications were predicted and actually correct.
  ![](/Images/Easy%20Ensemble%20AdaBoost%20Acc_Score.png)

-Precision: 0.09 of high risk applications were predicted and actually correct.
- Recall: 0.92 of actual high risk applications identified correctly.
  ![](/Images/Easy%20Ensemble%20AdaBoost%20Class_Report.png)

##  Summary

While some of the above machine learning models outperform others, I recommend more research be done to identify machine learning models that yield better prediction success.

However, in a scenario where limited to the options above, I'd recommend the Easy Ensemble model. Each of its scores reveal it most likely to accurately identify and predict high risk loan applications.

It is important to note that its F1 score of 0.16 is significantly higher than the other models as highlighted in the Classification Reports. F1 score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0.

There's usually a trade-off between sensitivity and precision and a balance must be struck between the two. A useful way to think about the F1 score is that a pronounced imbalance between sensitivity and precision will yield a low F1 score. While 0.16 is low, it is up to 8 times higher than the other models in this case. Hence, of the options above, most favored to predict high risk loan applications.
___
![](/Images/Back.jpeg)

```Denis Antonov 10.24.2022```

```Contact: antonov.resu@gmail.com```