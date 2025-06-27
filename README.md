# Logistic Regression Classifier - Breast Cancer Detection

## 🧠 AI & ML Internship - Task 4: Classification with Logistic Regression

This project involves building a *binary classification model* using *Logistic Regression* to predict whether a tumor is *malignant (M)* or *benign (B)* based on features extracted from cell nuclei in breast cancer images.

---

## 📁 Dataset

- *Name:* Breast Cancer Wisconsin (Diagnostic) Dataset
- *Source:* [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- *Target Column:* diagnosis (M = Malignant, B = Benign)

---

## 🔧 Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 🔍 Workflow

1. *Data Cleaning*
   - Dropped irrelevant columns: id, Unnamed: 32
   - Encoded target: M → 1, B → 0

2. *Preprocessing*
   - Split into training and test sets (80/20)
   - Standardized feature values using StandardScaler

3. *Modeling*
   - Trained a Logistic Regression model

4. *Evaluation*
   - *Confusion Matrix*
   - *Precision & Recall*
   - *Classification Report*
   - *ROC-AUC Score*
   - *ROC Curve Plot*
   - *Sigmoid Function Visualization*

---

## 📈 Evaluation Metrics

- *Precision:* Measures accuracy of positive predictions.
- *Recall:* Measures ability to find all positive cases.
- *ROC-AUC:* Measures the ability to distinguish classes.
- *Confusion Matrix:* Summarizes prediction results.

---

## 📊 Visualizations

- *Sigmoid Function Curve*
- *ROC Curve*

---

## 🤔 Interview Questions You Should Know

1. How does logistic regression differ from linear regression?
2. What is the sigmoid function?
3. What is precision vs recall?
4. What is the ROC-AUC curve?
5. What is the confusion matrix?
6. How do you handle imbalanced data?
7. How do you choose the threshold?
8. Can logistic regression be used for multi-class problems?

---

## 📎 Files Included

- data.csv - Dataset
- logistic_regression_classifier.py - Main script
- README.md - Project overview

---

## ✅ Submission

Upload all files to your GitHub repository and submit the repository link using the provided internship form.
