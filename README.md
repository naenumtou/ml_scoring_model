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
│   └── ...
├── src/
│   ├── create_factor.py
│   ├── modified_sampling.py
│   ├── features_prep.py
│   ├── features_selection.py
│   ├── mixed_matrix.py
│   ├── cluster_analysis.py
│   ├── model_builder.py
│   ├── score_construct.py
│   └── ...
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
#### 4.2 Features Selection
#### 4.3 Cluster Analysis
#### 4.4 Pilot Model
#### 4.5 Training Model

### 5. Score Development
#### 5.1 Optimized Base Odds and Point of Double Odds (PDO)


### 6. Result
...

## License
MIT · Built for learning purposes
