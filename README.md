# Diabetes Prediction using Machine Learning

This project predicts the likelihood of diabetes in patients based on medical diagnostic measurements. The analysis is performed using Python and multiple machine learning models, with steps covering EDA, preprocessing, feature engineering, model tuning, and performance evaluation.

## Dataset

- **Source**: Pima Indians Diabetes Dataset
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target**: `Outcome` (1 = Diabetic, 0 = Non-diabetic)

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualizations: histograms, distribution plots, pie chart of target classes
- Correlation matrix heatmap
- Class imbalance check

### 2. Data Preprocessing
- Missing values replaced with class-based medians
- Outlier detection and suppression using IQR and LOF

### 3. Feature Engineering
- New features created based on:
  - BMI ranges (e.g., Underweight, Normal, Obesity)
  - Insulin levels (Normal/Abnormal)
  - Glucose levels (Low, Normal, High)
- One-hot encoding applied to categorical variables

### 4. Feature Scaling
- Numeric features scaled using `RobustScaler` for better model performance

### 5. Model Training
- Base models evaluated using 10-fold cross-validation:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
  - LightGBM

### 6. Model Tuning
- Hyperparameter tuning using `GridSearchCV` for:
  - Random Forest
  - Gradient Boosting
  - LightGBM
- Feature importance visualized for interpretability

### 7. Model Comparison
- Accuracy scores compared using boxplots
- Final selected models demonstrate high precision and robustness

## Results

| Model               | Tuned | Accuracy (CV) |
|---------------------|--------|----------------|
| Random Forest       | Yes    | ~87.5%         |
| LightGBM            | Yes    | ~89.2%         |
| Gradient Boosting   | Yes    | ~89.5%         |

## Installation

Install the required Python libraries with:

pip install numpy pandas seaborn matplotlib scikit-learn lightgbm statsmodels missingno
