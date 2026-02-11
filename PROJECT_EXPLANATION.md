# Churn Prediction with SHAP: A Comprehensive Project Guide

This document provides a detailed explanation of the **Churn Prediction with SHAP** project. It covers the project's purpose, architecture, capabilities, and the underlying data science concepts used in its implementation.

## Project Overview

This project is a modular, object-oriented machine learning pipeline designed to predict customer churn in the telecommunications industry using the **IBM Telco Customer Churn Dataset**. It goes beyond simple prediction by integrating **SHAP (SHapley Additive exPlanations)** to provide deep insights into *why* a customer is predicted to churn.

The project is structured to be:
*   **Modular**: Key functionalities are encapsulated in separate Python files within the `src/` directory.
*   **Scalable**: The object-oriented design allows for easy addition of new models, feature engineering steps, or visualizations.
*   **Reproducible**: Random seeds and stratified splitting ensure consistent results across runs.

---

## Project Structure & Capabilities

The project is organized into three main layers:
1.  **Data Processing & Visualization** (`data_viz.py`)
2.  **Feature Engineering & Selection** (`feature_eng.py`)
3.  **Modeling & Evaluation** (`ml_model.py`, `exp.py`)

### 1. Exploratory Data Analysis (EDA)
**File**: `src/data_viz.py`
**Class**: `EDA`

The `EDA` class provides a suite of methods to automate the initial analysis of the dataset.

*   **Data Quality Checks**:
    *   **Missing Values**: Detects and visualizes missing data distributions (e.g., histograms or count plots of missing data against other features).
    *   **Duplicates**: Identifies and removes duplicate records.
    *   **Basic Statistics**: Generates summary statistics (mean, median, std) and data types.

*   **Univariate Analysis**:
    *   **Categorical Features**: Automatically generates count plots with percentage labels to show the distribution of categories (e.g., PaymentMethod, Contract).
    *   **Numerical Features**: Creates dual plots (KDE Plot + Box Plot) to show the distribution shape and detect outliers (e.g., MonthlyCharges, Tenure).

*   **Bivariate Analysis**:
    *   **Correlation Matrix**: Computes and visualizes the Pearson correlation between numerical features.
    *   **Crosstabulation**: Analyzes the relationship between two categorical variables (e.g., Churn vs. Contract Type).
    *   **Feature vs. Target**: Visualizes how a feature relates to the target variable (`Churn`) using grouped count plots (categorical) or box plots (numerical).

*   **Geospatial Analysis**:
    *   Uses **GeoPandas** and **Contextily** to plot customer locations (Latitude/Longitude) on a map of California, verifying the geographical consistency of the data.

### 2. Feature Engineering
**File**: `src/feature_eng.py`

This module transforms raw data into a format suitable for machine learning models.

*   **Encoding**:
    *   **One-Hot Encoding**: Converts nominal categorical variables (e.g., Gender, PaymentMethod) into binary vectors.
    *   **Label Encoding**: Converts the target variable (`Churn`) into 0 and 1.

*   **Scaling**:
    *   **Standard Scaler**: Standardizes numerical features by removing the mean and scaling to unit variance (Z-score normalization).
    *   **MinMax Scaler**: Scales features to a given range, typically [0, 1].

*   **Feature Selection**:
    *   **Filter Methods**:
        *   **ANOVA (Analysis of Variance)**: Selects numerical features based on their relationship with the categorical target.
        *   **Chi-Squared Test**: Selects categorical features based on their independence from the target.
    *   **Wrapper Methods**:
        *   **BorutaShap**: An advanced feature selection algorithm that uses SHAP values to find all relevant features, not just the strongest ones. It iteratively compares importance against "shadow features" (randomized versions of original features).

### 3. Machine Learning Pipeline
**File**: `src/ml_model.py`
**Class**: `ModelPipeline`

This class manages the training, evaluation, and comparison of multiple machine learning models.

*   **Supported Models**:
    *   **Logistic Regression**: A baseline linear model.
    *   **Random Forest**: An ensemble of decision trees.
    *   **XGBoost**: A powerful gradient boosting algorithm.
    *   **CatBoost**: A gradient boosting algorithm optimized for categorical data (handled natively).

*   **Imbalanced Data Handling**:
    *   **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic examples for the minority class (Churners) to balance the dataset.
    *   **ADASYN (Adaptive Synthetic Sampling)**: Similar to SMOTE but focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier.
    *   **KMeansSMOTE**: Applies KMeans clustering before over-sampling to generate safer synthetic samples.

*   **Evaluation Strategy**:
    *   **Stratified K-Fold Cross-Validation**: Splits the data into `k` folds while preserving the percentage of samples for each class. This ensures robust evaluation, especially for imbalanced datasets.
    *   **Metrics**: Calculates Accuracy, Recall, Precision, F1-Score, Balanced Accuracy, and ROC-AUC.

*   **Visualization**:
    *   **Violin Plots**: Compares the distribution of performance metrics across folds for different models.
    *   **ROC Curves**: Plots the Receiver Operating Characteristic curve to visualize the trade-off between True Positive Rate and False Positive Rate.

### 4. Explainability (SHAP)
**File**: `src/exp.py`

This module focuses on interpreting the "black box" models.

*   **Shapley Values**: Computes SHAP values, which represent the contribution of each feature to the prediction of a specific instance.
*   **Cross-Validation Integration**: Calculates SHAP values across all folds of the cross-validation to ensure the explanations are robust and not biased by a single train-test split.
*   **Global & Local Interpretability**:
    *   **Global**: Feature importance rankings, summary plots (beeswarm).
    *   **Local**: Force plots and waterfall plots to explain why a *specific* customer was predicted to churn.

---

## Detailed Concepts Explained

### Object-Oriented Programming (OOP) in Data Science
The project uses OOP to encapsulate logic. Instead of loose scripts, we have classes (`EDA`, `ModelPipeline`).
*   **Encapsulation**: Data (the dataframe) and methods (plotting, training) are bundled together.
*   **Reusability**: The `ModelPipeline` can be reused for different datasets or different sets of models without rewriting the training loop.

### Stratified K-Fold Cross-Validation
Standard cross-validation might split the data such that one fold has very few churn cases. **Stratified** K-Fold ensures that each fold has the same proportion of churners as the original dataset (e.g., 26%). This leads to a more reliable estimate of model performance.

### SMOTE & Imbalanced Learning
Churn datasets are typically imbalanced (e.g., 74% Stay, 26% Churn). A model could achieve 74% accuracy by simply predicting "Stay" for everyone.
*   **SMOTE** creates synthetic examples between existing minority class instances.
*   **Impact**: This forces the model to learn the decision boundary for churners more effectively, typically improving **Recall** (capturing more actual churners) at the cost of some Precision.

### SHAP (SHapley Additive exPlanations)
Based on cooperative game theory, SHAP assigns a value to each feature for each prediction.
*   **Positive SHAP value**: The feature pushes the prediction towards "Churn".
*   **Negative SHAP value**: The feature pushes the prediction towards "Stay".
*   **Magnitude**: The absolute size indicates the strength of the influence.
*   **Property**: The sum of SHAP values + the base value (average prediction) equals the actual model output. This guarantees **consistency** and **local accuracy**.

### BorutaShap
Standard feature selection (like taking the top 10 features from a Random Forest) can be biased. BorutaShap:
1.  Creates "Shadow Features" (shuffled copies of real features).
2.  Trains a model (XGBoost/Random Forest).
3.  Compares the importance of real features to the maximum importance of shadow features.
4.  Features significantly better than the best shadow feature are "Confirmed".
5.  This finds **all** relevant features, not just the top ones, reducing noise and overfitting.
