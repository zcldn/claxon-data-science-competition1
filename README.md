# Loan Default Prediction Model

For detailed API documentation, please refer to [API Documentation](API_DOCUMENTATION.md).

## Introduction
This project is designed to predict loan default status using a dataset of loan applications. The goal is to develop a robust machine learning model that can accurately predict whether a loan will be defaulted based on various features such as loan amount, interest rate, salary, and more. The model will be trained using a combination of feature engineering, feature selection, and machine learning algorithms to achieve high accuracy and reliability. The final model will be evaluated using various metrics and will be tested for bias mitigation to ensure fairness and equality in predictions.


## Final Model Score
- **ROC AUC Score:** 0.922 (92.2%)

## Table of Contents
1. [Final Model Score](#final-model-score)
2. [Project Overview](#project-overview)
3. [Initial Data Examination and Handling](#initial-data-examination-and-handling)
    - [Class Imbalance](#class-imbalance)
    - [Data Quality Issues](#data-quality-issues)
        - [Data Errors](#data-errors)
        - [Missing Values](#missing-values)
4. [Exploratory Data Analysis (EDA) Key Findings](#exploratory-data-analysis-eda-key-findings)
    - [Univariate Analysis](#univariate-analysis)
    - [Bivariate Relationships](#bivariate-relationships)
    - [Multivariate Correlations](#multivariate-correlations)
5. [Feature Engineering](#feature-engineering)
6. [Feature Selection](#feature-selection)
7. [Model Training](#model-training)
8. [Model Evaluation and Bias Mitigation](#model-evaluation-and-bias-mitigation)
9. [Data Drift Detection](#data-drift-detection)
10. [Final Pipeline](#final-pipeline)
11. [Additional Resources](#additional-resources)



**Explanation:**  
- Achieved using the CatBoost algorithm, indicating high model reliability and robustness.
- Score reflects the model’s accuracy in predicting loan defaults.
- Ensured consistency with 5-fold cross-validation.

## Project Overview
This project aims to predict loan default status using features such as loan amount, interest rate, salary, and more. The model leverages advanced machine learning techniques and thorough data preprocessing to achieve high accuracy.

## Initial Data Examination and Handling

### Class Imbalance
**Observation:**  
- The dataset exhibited significant class imbalance, which is common in loan approval problems.

**Handling:**  
- Applied resampling methods and adjusted class weights. These methods did not significantly improve the model performance.

### Data Quality Issues
**Data Errors:**  
- **Spelling Errors:** Numerous instances of incorrect spellings were found.
- **Trailing White Spaces:** Some fields contained unnecessary trailing white spaces.
- **Mixed Data Types:** Several columns had mixed data types requiring standardization.

**Missing Values:**  
- **Job and Location:** Missing values needed addressing.
- **Country:** This column had only one value, "Zimbabwe," making its missing values less critical.

### Data Cleaning
1. **Class Imbalance:**  
   - Implemented resampling and class weight adjustments.

2. **Data Quality Improvements:**  
   - Corrected spelling errors.
   - Removed trailing white spaces.
   - Standardized mixed data types.

3. **Missing Values:**  
   - Imputed `Job` values with the median job ("Data Analyst") due to stable distribution.
   - Imputed `Location` values with the mode.
   - Dropped the `Country` column.

## Exploratory Data Analysis (EDA) Key Findings

### Univariate Analysis
- **Non-normal Distributions:** Loan amounts, salaries, and outstanding balances were not normally distributed.
- **High-Density Loan Amount:** A notable peak around $5,000 indicated this was a common loan size.

### Bivariate Relationships
- **Geographic Impact:** Higher default probabilities in towns like Chiredzi, Shurugwi, and others compared to cities like Harare and Bulawayo.
- **Marital Status and Default:** Married individuals were less likely to default.
- **Occupation and Default:** Lawyers and accountants showed higher default probabilities.

### Multivariate Correlations
- **Strong Positive Correlations:**
  - Salary and loan amount (0.54)
  - Salary and remaining loan term (0.72)
  - Loan amount and outstanding balance (0.56)

## Feature Engineering
- **Custom Encoder:** Converted `Location` and `Job` into ordinal representations.
- **New Features Created:**
  - `Total_amount_per_location`
  - `Total_amount_per_job`
  - `Under_5000` (Boolean feature for loans around $5,000)
  - `Greater_than_75000` (Boolean feature for loans above $75,000)
  - Separated locations into Industrial Areas, Cities, and Others.
  - `Job_location_interact` to capture job-location correlation.

- **Marital Status:** Trained an XGBoost model to predict missing marital status values with 90% accuracy.

## Feature Selection
- Used variance threshold, random forest RFE, and chi-square tests to remove less significant features, including `Country` and `Currency`.

## Model Training
- Split data into train and test sets.
- Trained 7 models using 5-fold cross-validation. CatBoost had the highest score of 0.90.
- Evaluated on separate splits. Ensemble methods for specific loan amounts were ineffective.

## Model Evaluation and Bias Mitigation
**Evaluation:**
- Feature importances were balanced.
- Confusion matrix showed correct classifications.
- Sensitivity analysis detected bias in `Location` against Hwange and unknown locations.

**Bias Mitigation:**
- Created new features from important ones, improving the score to 0.922.
- Resampling techniques were ineffective.

## Data Drift Detection
- Used KS Test for numerical features and Population Stability for both categorical and numerical features.
- Detected drift in `Remaining Term` and `Outstanding Balance`.

## Final Pipeline
- Custom `DataTransformer` class for feature engineering.
- Data drift detection class.
- Final model integrated into the pipeline.

## Additional Resources
For detailed API documentation, please refer to [API Documentation](API_DOCUMENTATION.md).
