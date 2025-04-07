# Credit Risk Prediction Using Machine Learning

This project focuses on building a robust classification pipeline to predict the risk of loan default based on customer financial and demographic data.

## 📌 Objective

The goal of this project is to use machine learning techniques to assess the likelihood of loan default, helping financial institutions make informed lending decisions.

---

## 📊 Dataset

The dataset includes various features like:

- Customer demographics (age, gender, marital status)
- Financial indicators (income, bank balance, credit utilization)
- Loan-specific features (loan amount, purpose, tenure, sanction amount)

The target variable is `default`, indicating whether a customer defaulted on their loan.

---

## ⚙️ Workflow

### 1. **Data Preprocessing**
- Handling missing values
- Feature engineering (e.g., loan-to-income ratio)
- Outlier detection
- Label encoding and scaling (MinMaxScaler)

### 2. **Exploratory Data Analysis (EDA)**
- Histograms and bar charts for all numerical and categorical columns
- Correlation analysis

### 3. **Information Value (IV) & WOE Encoding**
- IV calculation for feature selection
- Binning continuous variables
- Filtering features with IV > 0.02

### 4. **Model Building**
- Applied multiple classifiers: `DecisionTreeClassifier`, `SVC`, `LogisticRegression`
- Hyperparameter tuning using:
  - `Optuna`
  - `RandomizedSearchCV`

### 5. **Model Evaluation**
- Accuracy, precision, recall, F1-score
- Confusion matrix
- ROC curve and AUC

---

## 🧠 Best Performing Model

The project identifies the best-performing model based on cross-validation scores and provides detailed analysis of model performance.

---

## 📈 Technologies Used

- Python
- Pandas, NumPy
