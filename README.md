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
This project builds a **scoring model** using supervised ML to predict a target score (e.g. default risk or probability of default).
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
│   └── ...
├── data/         #Not tracked by git
├── requirements.txt
└── README.md
```

## Project Details
...

## License
MIT · Built for learning purposes
