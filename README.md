# 🩺 Diabetes Prediction Using Machine Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-189A4D?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<br>

> **An end-to-end Machine Learning project** that predicts diabetes risk using clinical and demographic data.  
> Covers the complete ML lifecycle — from raw data to a fully deployed Streamlit web application.

<br>

![Diabetes Prediction Banner](https://img.shields.io/badge/🩺_Diabetes_Risk_Prediction_System-AI_Powered_Clinical_Screening-0f3460?style=for-the-badge)

</div>

---

## 📋 Table of Contents
<!-- - [Live Demo](#-live-demo) -->
- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Installation & Usage](#-installation--usage)
- [How to Use the App](#️-how-to-use-the-app)
- [Test Cases](#-quick-test-cases)
- [Tech Stack](#️-tech-stack)
- [Future Work](#-future-work)
- [Disclaimer](#️-disclaimer)
- [Author](#-author)

---

## 🎯 Project Overview

Diabetes is one of the leading chronic diseases globally, affecting over **537 million adults** worldwide. Early prediction and diagnosis can significantly reduce healthcare costs and improve patient outcomes.

This project builds a **binary classification model** to predict whether a patient is diabetic or not, based on 8 clinical and demographic features. The model is served through a professional **Streamlit web application** with real-time risk categorisation.

---

<!-- ## 🌐 Live Demo

> 🚀 **[Click here to try the app](#)** ← *(Add your Streamlit Cloud URL here after deployment)*

```
streamlit run app.py → http://localhost:8501
```

--- -->

## 🏆 Key Results

<div align="center">

| Metric | Score |
|--------|-------|
| 🎯 ROC-AUC | **0.977** |
| 📊 F1 Score | **0.87** |
| 🔍 Recall | **0.86** |
| ✅ Accuracy | **0.97** |
| 📦 Training Samples | **100,000** |
| 🤖 Models Compared | **11** |

</div>

---

## ✨ Features

### 🔬 Machine Learning
- ✅ Complete EDA with **20+ professional visualisations**
- ✅ **11 ML algorithms** trained and compared side by side
- ✅ Hyperparameter tuning with **RandomizedSearchCV** (30 iterations)
- ✅ **SHAP explainability** — global summary, bar plot, and per-prediction waterfall
- ✅ **SMOTE** for class imbalance handling (8.5% diabetic minority class)
- ✅ **Voting & Stacking ensemble** classifiers
- ✅ Calibration curves and learning curves
- ✅ Permutation importance analysis
- ✅ Error analysis on misclassified samples
- ✅ 5-fold stratified cross-validation

### 🖥️ Web Application
- ✅ Beautiful **Streamlit UI** with custom CSS styling
- ✅ Real-time **probability score** with animated progress bar
- ✅ **Three-tier risk categorisation** — 🟢 Low / 🟡 Medium / 🔴 High
- ✅ Patient summary table with clinical reference ranges
- ✅ Key risk factor identification per prediction
- ✅ Debug info expander for transparency
- ✅ Fully responsive sidebar input form

---

## 📁 Project Structure

```
diabetes-prediction-ml/
│
├── 📓 notebook.ipynb                    # Complete ML notebook (17 sections)
├── 🖥️  app.py                            # Streamlit web application
├── 📋 requirements.txt                  # All Python dependencies
├── 📄 README.md                         # This file
├── 📊 diabetes_prediction_dataset.csv   # Dataset (100,000 records)
│
└── 📦 model_artifacts/                  # Generated after running notebook
    ├── model.pkl                        # Trained XGBoost model
    ├── scaler.pkl                       # Fitted StandardScaler
    ├── encoder.pkl                      # Label encoder for gender
    └── top_features.pkl                 # Selected feature names list
```

---

## 🔄 ML Pipeline

```
Raw Data (100,000 records)
        │
        ▼
Data Cleaning & EDA
(duplicates, outliers, distributions, correlations)
        │
        ▼
Feature Engineering
(label encoding, one-hot encoding, StandardScaler)
        │
        ▼
Feature Selection
(Mutual Information + RFE → top 8 features)
        │
        ▼
Train / Validation / Test Split
(70% / 15% / 15% — stratified)
        │
        ▼
11 Models Trained & Compared
(LR, DT, RF, ET, KNN, NB, SVM, GBM, XGBoost, LightGBM, CatBoost)
        │
        ▼
Hyperparameter Tuning
(RandomizedSearchCV — XGBoost)
        │
        ▼
SHAP Explainability + Ensemble Methods
        │
        ▼
Model Saved → Streamlit App Deployed
```

---

## 📊 Model Performance

### All Models Comparison (Validation Set)

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 | **XGBoost (Tuned)** | **0.974** | **0.891** | **0.863** | **0.877** | **0.977** |
| 🥈 | LightGBM | 0.972 | 0.884 | 0.851 | 0.867 | 0.974 |
| 🥉 | CatBoost | 0.971 | 0.880 | 0.848 | 0.864 | 0.973 |
| 4 | Extra Trees | 0.969 | 0.872 | 0.840 | 0.856 | 0.969 |
| 5 | Random Forest | 0.968 | 0.868 | 0.835 | 0.851 | 0.967 |
| 6 | Gradient Boosting | 0.965 | 0.858 | 0.822 | 0.840 | 0.963 |
| 7 | SVM | 0.958 | 0.841 | 0.798 | 0.819 | 0.951 |
| 8 | Logistic Regression | 0.952 | 0.820 | 0.775 | 0.797 | 0.924 |
| 9 | KNN | 0.948 | 0.808 | 0.761 | 0.784 | 0.918 |
| 10 | Decision Tree | 0.939 | 0.782 | 0.738 | 0.759 | 0.867 |
| 11 | Naive Bayes | 0.921 | 0.731 | 0.695 | 0.713 | 0.912 |

### Top Features (SHAP Analysis)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | 🩸 HbA1c Level | Highest |
| 2 | 🍬 Blood Glucose Level | Very High |
| 3 | 🎂 Age | High |
| 4 | ⚖️ BMI | Moderate |
| 5 | 💊 Hypertension | Moderate |
| 6 | 🚬 Smoking (No Info) | Low |
| 7 | ❤️ Heart Disease | Low |
| 8 | 🚬 Smoking (Former) | Low |

---

## 🔬 Dataset

| Property | Value |
|----------|-------|
| Source | Diabetes Prediction Dataset (Kaggle) |
| Total Records | 100,000 patients |
| Features | 9 (age, gender, BMI, HbA1c, blood glucose, hypertension, heart disease, smoking history) |
| Target | Binary (0 = Non-Diabetic, 1 = Diabetic) |
| Class Distribution | 91.5% Non-Diabetic / 8.5% Diabetic |
| Missing Values | None |
| Duplicates | Removed during cleaning |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/JiteshAnand07/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Create Virtual Environment
```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebook
```bash
jupyter notebook notebook.ipynb
```
Execute all cells top to bottom. This generates the `model_artifacts/` folder automatically. Takes approximately 5–10 minutes.

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ How to Use the App

1. Fill in patient details in the **left sidebar**
2. Adjust sliders for Age, BMI, HbA1c Level, and Blood Glucose
3. Check boxes for Hypertension and Heart Disease if present
4. Select Smoking History from the dropdown
5. Click **"🔍 Predict Diabetes Risk"**
6. View the probability score, risk category, and key risk factors

---

## 🧪 Quick Test Cases

| | 🟢 **Low Risk** | 🟡 **Medium Risk** | 🔴 **High Risk** |
|---|---|---|---|
| **Profile** | Healthy Young Female | Middle-Aged Male | Older Male (Multiple Risk Factors) |
| **Gender** | Female | Male | Male |
| **Age** | 35 | 45 | 55 |
| **BMI** | 22.0 | 28.0 | 32.0 |
| **HbA1c** | 5.0% | **6.8%** | 7.5% |
| **Glucose** | 90 mg/dL | **100 mg/dL** | 200 mg/dL |
| **Hypertension** | ❌ | ❌ | ✅ |
| **Heart Disease** | ❌ | ❌ | ❌ |
| **Smoking** | never | never | current |
| **Expected Result** | 🟢 ~0–5% probability | 🟡 ~51% probability | 🔴 ~95–99% probability |
---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| ML & Data | Scikit-learn, XGBoost, LightGBM, CatBoost, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Explainability | SHAP |
| Imbalanced Data | Imbalanced-learn (SMOTE) |
| Model Persistence | Joblib |
| Web App | Streamlit |
| Notebook | Jupyter |
| Version Control | Git & GitHub |

---

## 🔮 Future Work

- 🌐 Deploy on Streamlit Cloud with public URL
- 📄 PDF report export per prediction
- 🤖 AI health advice using OpenAI API
- 🗄️ SQLite patient history database
- 📱 Mobile app (React Native)
- 📡 Data drift monitoring with Evidently AI

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**.  
It is **not** intended for clinical use or actual medical decision-making.  
Always consult a qualified healthcare professional for medical advice.

---

## 👤 Author

**Jitesh Anand**  
Machine Learning & Data Science Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/JiteshAnand07)

---

<div align="center">

⭐ **If you found this project useful, please star the repository!** ⭐

</div>
