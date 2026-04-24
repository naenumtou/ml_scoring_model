# ML Scoring Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-success?logo=lightgbm&style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow?logo=catboost&style=for-the-badge)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Machine Learning–based scoring model for risk / propensity / credit-style scoring.
Designed for real-world usage with explainability, stability, and production readiness in mind.

<img width="1375" height="768" alt="B-Score model แบบใช้ Machine learning model ในการพัฒนา" src="https://github.com/user-attachments/assets/96d8358d-a0fb-40fc-92df-e6bb6e4d8848" />

## Overview
A Behavioral Score (B-Score) is a statistical model used to assess the credit risk of existing customers based on their historical behavior with a financial institution. Unlike an Application Score (A-Score), which uses "static" data from the time of enrollment, the B-Score is dynamic, evolving as the customer interacts with the product.

This project builds a **B-Scoring model** using supervised ML to predict a target score (e.g. default risk or probability of default).
The workflow follows industry-standard model development practices.

## Project Structure
```
ml_scoring_model/
├── models/       #Trainned model (.cbm) and study parameters (pkl.)
├── notebooks/
│   ├── 01_factor_creation.ipynb
│   ├── 02_data_sampling.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_score_evaluation.ipynb
├── src/
│   ├── create_factor.py
│   ├── modified_sampling.py
│   ├── features_prep.py
│   ├── features_selection.py
│   ├── mixed_matrix.py
│   ├── cluster_analysis.py
│   ├── model_builder.py
│   ├── score_construct.py
│   └── back_testing.py
├── data/         #Not tracked by git
├── requirements.txt
└── README.md
```

## Project Details
### 1. Overview of Behavioral Factors
Raw transaction data is often too granular and noisy for machine learning models. Factor Engineering (or Feature Engineering) is the process of transforming raw logs into meaningful predictors. In this project, factors are generally categorized into the following aspects:

| No. | Aspect | Description |
|------|--------|-------------|
| 1 | Account balance | The total amount of money remaining in a customer’s account (or the total debt owed on a credit line) at a specific point in time. |
| 2 | Due amount | The portion of the total balance that must be paid—including the principal, interest, and fees within the current billing cycle. |
| 3 | Repayment | The act of a borrower paying back the principal and interest on a loan or credit facility. |
| 4 | Delinquency status | A snapshot of how many days a payment is overdue. It is typically measured in Days Past Due (DPD), categorized into buckets. |

### 2. Factors Creation
The primary reason for creating behavioral factors is to transform massive, noisy transaction data into dynamic insights that reflect a customer's current financial health. Unlike static application data, behavioral factors capture trends such as shifting spending patterns, declining repayment habits, or increasing credit reliance that allowing the model to detect "early warning signs" months before an actual default occurs.

<img width="1408" height="768" alt="B-Score model แบบใช้ Machine learning model ในการพัฒนา" src="https://github.com/user-attachments/assets/70111913-c4cc-438c-9831-0bfac837c006" />

$~$

The example of behavioral factors are listed below:
| No. | Factor | Description |
|------|--------|-------------|
| 1 | avg_bal_3 | The average of account balance over the last 3 months. |
| 2 | max_due_3_to_fin | The maximum due amount over the last 3 months to initial financial amount. |
| 3 | n_fully_pay_3 | The number of months (times) that fully payment made over the last 3 months. |
| 4 | max_del_3 | The maximum delinquency over the last 3 months |

### 3. Modified Train/Test Split
Unlike a standard train/test split even one stratified on the default rate, the behavioral data is designed to capture granular on transaction level patterns over time. A traditional random split at the record level can lead to data leakage, where a single customer’s history is fragmented across both training and testing sets, resulting in an over optimistic model.


To mitigate this, the modified train/test split of customer level partitioning strategy is implemented to ensure that all records belonging to a specific customer are confined to only one dataset. Furthermore, this split was meticulously balanced to minimize the variance in default rates across both the global population and on a month by month basis, ensuring the model's stability and performance remain consistent over time.

<img width="1376" height="768" alt="B-Score model แบบใช้ Machine learning model ในการพัฒนา" src="https://github.com/user-attachments/assets/039185c8-1e7e-47aa-ab4c-ef9b5e921320" />

### 4. Model Development
#### 4.1 Features Preparation
The process transforms categorical labels into numerical values using K-Fold Target Encoding. Instead of just calculating a simple average of the target for each category, it uses cross-validation to ensure that the value assigned to a row is calculated from other data "folds." This prevents the model from "cheating" (data leakage) and reduces overfitting. It also uses "smoothing" to balance category means with the global average and smartly fills any missing or new categories with the overall mean.

The process then cleans up numerical data by filling in missing values using MICE (Multiple Imputation by Chained Equations). Rather than just plugging in a static mean or median, it treats every missing value as a target to be predicted by a Bayesian Ridge model based on other available features. It also automatically converts infinite values to nulls so they can be imputed properly, ensuring your final dataset is complete, statistically sound, and ready for feature selection.

<img width="1408" height="620" alt="B-Score model แบบใช้ Machine learning model ในการพัฒนา" src="https://github.com/user-attachments/assets/d8674e5f-5354-4fbf-94ed-4b0033a08f60" />


#### 4.2 Features Selection
The automated feature selection is performed by creating a `competition` between your real data and randomized noise. First, it generates shadow features by shuffling the values of your original data to destroy any actual relationship with the target. It then trains a `LightGBM` model on both the real and shadow features combined, measuring their gain importance. The function calculates a selection threshold based on the average performance of these shadow features (scaled by threshold adjustment parameter). Finally, it filters out any original features that did not perform significantly better than the randomized shadows.

<img width="1408" height="768" alt="B-Score model แบบใช้ Machine learning model ในการพัฒนา" src="https://github.com/user-attachments/assets/f98d0cea-b77f-424d-b0a7-084abd181094" />

#### 4.3 Cluster Analysis and Pilot Model
This step represents an advanced feature selection pipeline designed to eliminate data redundancy using a combination of statistical clustering and machine learning. It follows a 3 steps logical flow:

1. Grouping Redundant Variables (Hierarchical Clustering): The cluster analysis calculates a "Distance Matrix" based on feature correlations. If features are highly correlated (e.g., > 70%), they are considered redundant and grouped into the same cluster.
2. Evaluating real impact (SHAP Importance): The shap model builds a quick "Pilot Model" using `CatBoost`. Instead of just looking at linear correlations, it uses **SHAP Importance** to measure how much each feature actually contributes to the model's predictions. This ensures that we know which variables are truly powerful and which are just noise.
3. Smart Representative Selection: The function is the final decision-maker. It looks at each cluster and picks the "Best Representative" based on two main criteria 1) Performance: It prioritizes the feature with the highest SHAP Score within its cluster. 2) Diversity: It ensures that different feature groups are represented, dropping the redundant "weaker" versions. As a result, the high performing list of features while dropping the redundant ones to prevent overfitting.

<img width="1408" height="768" alt="Gemini_Generated_Image_flxj91flxj91flxj" src="https://github.com/user-attachments/assets/a8f176e8-6e48-4cb3-84ce-7ec9fe9e1ee1" />


#### 4.4 Training Model

### 5. Score Development
#### 5.1 Optimized Base Odds and Point of Double Odds (PDO)


### 6. Result
...

## License
MIT · Built for learning purposes
