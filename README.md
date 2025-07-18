# Loan Eligibility Predictor using ML

A smart and interactive AI-powered web app built with Streamlit that predicts loan eligibility using machine learning. The app features real-time predictions, beautiful charts, voice feedback (local), and a modern UI.

---
## 🚀 Live App

[![Streamlit](https://img.shields.io/badge/Streamlit-App-%23FF4B4B?logo=streamlit&logoColor=white)](https://loan-eligibilty-predictor.streamlit.app/)

**Try it live here:** [https://loan-eligibilty-predictor.streamlit.app/](https://loan-eligibilty-predictor.streamlit.app/)
---

## 🚀 Overview

This project helps users assess loan eligibility using:
- ✅ **Random Forest Classifier** – Accuracy: 94.2%
- ✅ **Logistic Regression** – Accuracy: 91.8%

Users input personal and financial details, and the app instantly predicts eligibility along with confidence scores.

---

## 🔍 Features

- 🎯 Real-time loan eligibility prediction
- 📊 Income and loan amount visualization
- 🔁 Dual model selection (RF & LR)
- 📈 Evaluation charts: ROC curve and confusion matrices
- 🔊 Voice feedback (local use only with `pyttsx3`)
- 🖥️ Responsive, user-friendly UI

---

## 🧠 Models & Data

- **Dataset**: [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Records**: 614 entries, 13 features
- **Target**: Loan approval status (binary)

### Features Used:
- Age, Gender, Married, Dependents, Education, Self-Employed
- Applicant & Coapplicant Income, Loan Amount, Loan Term
- Credit History, Property Area

---

## 🧪 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Random Forest       | 94.2%    | 93.8%     | 94.5%  | 94.1%    |
| Logistic Regression | 91.8%    | 91.2%     | 92.1%  | 91.6%    |

---

## 📦 Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/loan-eligibility-predictor.git
cd loan-eligibility-predictor

# 2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
