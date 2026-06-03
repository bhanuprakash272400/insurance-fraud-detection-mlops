# 🛡️ Insurance Fraud Detection System — MLOps

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.4.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![MLflow](https://img.shields.io/badge/MLflow-2.12.1-blue)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Overview
An end-to-end ML-based insurance fraud detection system that identifies
suspicious claims using predictive analytics and classification models.
Built with Random Forest achieving **ROC-AUC of 0.9618**.

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.9618 ✅ Best** |
| Recall (Fraud) | 92.4% |
| Accuracy | 87.4% |
| F1 Score | 0.880 |
| Precision | 84.0% |

---

## 📊 Model Comparison

| Model | ROC-AUC | F1 |
|---|---|---|
| **Random Forest** | **0.962 ✅** | **0.880** |
| Gradient Boosting | 0.827 | 0.760 |
| SVM | 0.846 | 0.750 |
| Decision Tree | 0.783 | 0.750 |
| Logistic Regression | 0.730 | 0.680 |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9 |
| ML Models | Random Forest, Decision Tree, SVM, Gradient Boosting, Logistic Regression |
| ML Library | Scikit-Learn, Pandas, NumPy |
| Serving | FastAPI, Gradio |
| Experiment Tracking | MLflow |
| Deployment | Docker |

---

## ✨ Key Highlights
- 🔍 Compared **5 ML models** — Random Forest best
- 🧠 **26 engineered features** for fraud detection
- ⚡ Real-time fraud scoring via **FastAPI** (sub-200ms)
- 🖥️ **Gradio UI** for interactive predictions
- 📈 **MLflow** experiment tracking
- 📂 **Batch prediction** support (CSV upload, max 100 claims)
- 💡 **Explainability endpoint** — top 5 features driving each decision

---

## 📁 Project Structure

```
insurance-fraud-detection/
├── app.py                        # FastAPI inference API
├── gradio_app.py                 # Gradio UI for predictions
├── fraud_claims.ipynb            # Model training notebook
├── best_model.pkl                # Trained Random Forest model
├── best_model_features.json      # 26 feature names
├── best_model_metadata.json      # Model metrics & metadata
├── insurance_fraud_claims_sample.csv  # Sample dataset
├── insurance_fraud_cleaned.csv   # Cleaned dataset
├── fraud_ml_results.png          # Model comparison charts
├── model_comparison.csv          # All model metrics
└── requirements.txt              # Dependencies
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bhanuprakash272400/insurance-fraud-detection.git
cd insurance-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI server
```bash
uvicorn app:app --reload --port 8000
```
API live at: http://localhost:8000
Swagger docs: http://localhost:8000/docs

### 4. Run Gradio UI
```bash
python gradio_app.py
```
UI live at: http://localhost:7860

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check + model info |
| GET | `/features` | All 26 feature names |
| GET | `/model/info` | Full model metadata |
| POST | `/predict` | Single claim prediction |
| POST | `/predict/batch` | Batch prediction (max 100) |
| POST | `/predict/explain` | Prediction + top 5 feature explanation |

---

## 📝 Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "annual_premium": 25000.0,
    "policy_tenure_months": 36,
    "claim_amount": 180000.0,
    "num_claims_last_3yrs": 3,
    "vehicle_age": 2.0,
    "incident_severity": "Total Loss",
    "incident_type": "Fire",
    "police_report": "No",
    "witness_present": "No",
    "customer_state": "KA",
    "channel": "Online",
    "claim_status": "Open"
  }'
```

## 📝 Example Response

```json
{
  "claim_id": "CLM-12345",
  "prediction": {
    "fraud_predicted": true,
    "fraud_probability": 0.8732,
    "risk_level": "HIGH",
    "message": "Very likely fraud — flag for investigation"
  },
  "processed_at": "2024-01-01T10:00:00"
}
```

---

## 🎯 Risk Levels

| Risk Level | Probability | Action |
|---|---|---|
| **HIGH** | ≥ 75% | Flag for investigation |
| **MEDIUM** | 50-75% | Manual review recommended |
| **LOW** | 30-50% | Monitor |
| **VERY LOW** | < 30% | Approve |

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Domain** | Insurance Claims |
| **Total Samples** | 1,714 |
| **Train Samples** | 1,371 |
| **Test Samples** | 343 |
| **Features** | 26 |
| **Fraud Rate** | 28.6% |

---

## 🔬 Top Features Driving Fraud Detection

1. `claim_amount` — Highest importance
2. `claim_to_premium_ratio` — engineered feature
3. `annual_premium`
4. `age`
5. `policy_tenure_months`

---

## 📬 Contact

**Bhanu Prakash Theertham**
📧 bhanu.theertham@gmail.com
🔗 [github.com/bhanuprakash272400](https://github.com/bhanuprakash272400)
