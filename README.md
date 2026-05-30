# Insurance Fraud Detection System — MLOps

## Overview
ML-based fraud detection system to identify suspicious 
insurance claims using predictive analytics and 
classification models. Built with Random Forest achieving 
ROC-AUC of 0.9618.

## Tech Stack
- Python, Scikit-Learn, Pandas, NumPy
- FastAPI, Gradio, MLflow
- Random Forest, Decision Tree, SVM, 
  Gradient Boosting, Logistic Regression

## Model Performance
| Metric | Score |
|---|---|
| ROC-AUC | 0.9618 |
| Recall (Fraud) | 92.4% |
| Accuracy | 87.4% |
| F1 Score | 0.880 |

## Key Highlights
- Compared 5 ML models — Random Forest best
- 26 engineered features for fraud detection
- Real-time fraud scoring via FastAPI
- Gradio UI for interactive predictions
- MLflow experiment tracking

## Project Structure
├── app.py
├── gradio_app.py
├── best_model.pkl
├── fraud_claims.ipynb
├── requirements.txt
├── insurance_fraud_claims_sample.csv
└── insurance_fraud_cleaned.csv
