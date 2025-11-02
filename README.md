# PCA vs Bayesian Shrinkage vs OLS and PLS  
### Predicting Loan Approval through Dimensionality Reduction and Regularization

---

# Project Overview
This project was developed as part of the **Machine Learning for Economists** course (University of Bologna, 2025).  
It explores how different **linear prediction methods** perform in a high-dimensional setting, comparing:

- **Principal Component Analysis (PCA)**  
- **Bayesian Shrinkage methods** (Ridge and Lasso)  
- **Ordinary Least Squares (OLS)**  
- **Partial Least Squares (PLS)**  

The empirical application focuses on predicting **loan approval** using a dataset containing financial and demographic features of loan applicants.  
The analysis aims to evaluate how **dimensionality reduction** (PCA, PLS) and **regularization** (Ridge, Lasso) improve predictive accuracy and model stability compared to classical OLS.

---

# Objectives
- Implement and compare PCA, Ridge, Lasso, OLS, and PLS models.  
- Analyze the **bias–variance trade-off** and **multicollinearity** in high-dimensional data.  
- Evaluate model performance using **cross-validation**, **confusion matrices**, **ROC curves**, and **Mean Squared Error (MSE)**.  
- Connect empirical findings to the theoretical framework of *De Mol, Giannone & Reichlin (2008)*.

---

# Dataset
**Source:** Loan Approval Dataset (public dataset adapted for academic use)  
- Target variable: `loan_status` (Approved = 1, Rejected = 0)  
- Predictors: income, loan amount, credit score (CIBIL), asset values, employment type, and education.  
- Format: CSV file located in `/data/loan_approval_dataset.csv`

---

# Implementation
The full analysis is implemented in Python within a **Jupyter Notebook**:  
 [`Machine_Learning_4_Eco.ipynb`](Machine_Learning_4_Eco.ipynb)

Main libraries used:
- `pandas`, `numpy` — data manipulation  
- `scikit-learn` — model implementation (PCA, Ridge, Lasso, PLS, OLS)  
- `matplotlib`, `seaborn` — visualization  
- `scipy` — statistical testing  

The notebook includes:
1. Data preprocessing and standardization  
2. Model fitting and parameter tuning (cross-validation)  
3. Prediction and model validation  
4. Performance comparison and visual analysis  

---

# Results Summary
- Ridge and Lasso (Bayesian Shrinkage) achieved predictive performance similar to PCA, confirming the findings of *De Mol et al. (2008)*.  
- PCA and PLS reduced variance effectively but at the cost of interpretability.  
- OLS served as the baseline model and showed the highest variance and lowest out-of-sample performance.  
- Confusion matrices, ROC curves, and performance tables are available in the notebook output.

---

# Repository Structure

