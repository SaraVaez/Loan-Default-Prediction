# Loan Default Prediction Using Machine Learning

This project builds a machine learning pipeline to predict whether a borrower is likely to default on a loan. It uses real-world financial data and applies both Logistic Regression and Random Forest models. The final solution demonstrates practical use of data science in credit risk analysis.

## 📊 Project Objectives

- Clean and explore real-world financial data
- Engineer features and handle categorical variables
- Train and evaluate machine learning models
- Improve classification of defaulting borrowers
- Identify key predictors of loan default

## 📁 Dataset

- Source: Kaggle Loan Default Dataset  
- Size: 255,347 rows × 18 columns  
- Target Variable: `Default` (1 = default, 0 = no default)

## 🔍 Data Cleaning & Preparation

- Checked for missing values and duplicates  
- Used boxplots to identify outliers (none found in "Income")  
- Converted categorical features using **target encoding** to avoid high-dimensionality from one-hot encoding  
- Split data into training (80%) and testing (20%) sets  
- Standardized numeric variables

## 📊 Exploratory Data Analysis (EDA)

- Reviewed class balance: highly imbalanced toward non-defaults  
- Generated correlation heatmap  
- Observed weak linear correlations:
  - Slight negative correlation: Age vs. Default
  - Slight positive correlation: Loan Amount & Interest Rate vs. Default

## 🏗️ Feature Engineering

- Applied **target encoding** to categorical columns  
- Dropped unnecessary columns  
- Selected relevant numerical and encoded features

## 🧠 Models & Evaluation

### Logistic Regression
- High precision and recall for class 0 (non-default)
- Poor recall for class 1 (default): 0.01
- Overall weak at detecting defaults due to class imbalance

### Random Forest
- Improved classification performance
- Able to model nonlinear relationships
- Extracted **feature importance** to determine key drivers of default

## 📈 Results

| Model              | Precision (Default) | Recall (Default) | F1 Score (Default) |
|-------------------|---------------------|------------------|--------------------|
| Logistic Regression | 0.63                | 0.01             | 0.03               |
| Random Forest       | ✅ Higher recall     | ✅ Better balance | ✅ Better F1        |

> Random Forest outperformed Logistic Regression in identifying defaults.

## 🛠 Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest
- XGBoost (optional enhancement)
- Seaborn, Matplotlib

## 📦 Folder Structure



