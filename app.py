import pickle
import json
import numpy as np
import time
import os
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="""
## ML-powered API to detect fraudulent insurance claims

**Model**: Random Forest
**ROC-AUC**: 0.9618 | **Recall (Fraud)**: 92.4% | **F1**: 0.880

### Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | /health | Server health + model info |
| GET | /features | All 26 model features |
| POST | /predict | Single claim prediction |
| POST | /predict/batch | Bulk predictions (max 100) |
| POST | /predict/explain | Prediction + top feature explanation |
""",
    version="2.0.0",
)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = BASE

with open(os.path.join(ROOT, "best_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(ROOT, "best_model_features.json")) as f:
    FEATURES = json.load(f)
with open(os.path.join(ROOT, "best_model_metadata.json")) as f:
    METADATA = json.load(f)


class ClaimInput(BaseModel):
    age:                  int   = Field(..., ge=18, le=100,  examples=[35])
    annual_premium:       float = Field(..., gt=0,           examples=[25000.0])
    policy_tenure_months: int   = Field(..., ge=1,           examples=[36])
    claim_amount:         float = Field(..., gt=0,           examples=[180000.0])
    num_claims_last_3yrs: int   = Field(..., ge=0, le=20,    examples=[3])
    vehicle_age:          float = Field(..., ge=0,           examples=[2.0])
    incident_severity:    str   = Field(...,                 examples=["Total Loss"])
    incident_type:        str   = Field(...,                 examples=["Fire"])
    police_report:        str   = Field(...,                 examples=["No"])
    witness_present:      str   = Field(...,                 examples=["No"])
    customer_state:       str   = Field(...,                 examples=["KA"])
    channel:              str   = Field(...,                 examples=["Online"])
    claim_status:         str   = Field(...,                 examples=["Open"])

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }

    @field_validator("incident_severity")
    @classmethod
    def val_severity(cls, v):
        if v.strip().lower() not in ["minor", "major", "total loss"]:
            raise ValueError("Must be one of: Minor, Major, Total Loss")
        return v

    @field_validator("incident_type")
    @classmethod
    def val_incident(cls, v):
        if v.strip().lower() not in ["collision", "fire", "theft", "natural disaster"]:
            raise ValueError("Must be one of: Collision, Fire, Theft, Natural Disaster")
        return v

    @field_validator("police_report", "witness_present")
    @classmethod
    def val_yesno(cls, v):
        if v.strip().lower() not in ["yes", "no"]:
            raise ValueError("Must be Yes or No")
        return v

    @field_validator("channel")
    @classmethod
    def val_channel(cls, v):
        if v.strip().lower() not in ["agent", "branch", "online"]:
            raise ValueError("Must be one of: Agent, Branch, Online")
        return v

    @field_validator("claim_status")
    @classmethod
    def val_status(cls, v):
        if v.strip().lower() not in ["open", "closed", "under review"]:
            raise ValueError("Must be one of: Open, Closed, Under Review")
        return v

    @field_validator("customer_state")
    @classmethod
    def val_state(cls, v):
        if v.strip().upper() not in ["KA", "MH", "AP", "TS", "TN", "DL"]:
            raise ValueError("Must be one of: KA, MH, AP, TS, TN, DL")
        return v


class BatchInput(BaseModel):
    claims: List[ClaimInput] = Field(..., min_length=1, max_length=100)


def encode(claim: ClaimInput) -> np.ndarray:
    d = {k: str(v).strip().lower() for k, v in claim.model_dump().items()}
    severity_map = {"minor": 1, "major": 2, "total loss": 3}
    yn = {"yes": 1, "no": 0}
    row = {
        "age":                             claim.age,
        "annual_premium":                  claim.annual_premium,
        "policy_tenure_months":            claim.policy_tenure_months,
        "claim_amount":                    claim.claim_amount,
        "num_claims_last_3yrs":            claim.num_claims_last_3yrs,
        "vehicle_age":                     claim.vehicle_age,
        "incident_severity":               severity_map[d["incident_severity"]],
        "police_report":                   yn[d["police_report"]],
        "witness_present":                 yn[d["witness_present"]],
        "incident_type_fire":              int(d["incident_type"] == "fire"),
        "incident_type_natural disaster":  int(d["incident_type"] == "natural disaster"),
        "incident_type_theft":             int(d["incident_type"] == "theft"),
        "customer_state_dl":               int(d["customer_state"] == "dl"),
        "customer_state_ka":               int(d["customer_state"] == "ka"),
        "customer_state_mh":               int(d["customer_state"] == "mh"),
        "customer_state_tn":               int(d["customer_state"] == "tn"),
        "customer_state_ts":               int(d["customer_state"] == "ts"),
        "channel_branch":                  int(d["channel"] == "branch"),
        "channel_online":                  int(d["channel"] == "online"),
        "claim_status_open":               int(d["claim_status"] == "open"),
        "claim_status_under review":       int(d["claim_status"] == "under review"),
        "claim_to_premium_ratio":          claim.claim_amount / (claim.annual_premium + 1),
        "is_repeat_claimant":              int(claim.num_claims_last_3yrs >= 3),
        "tenure_bucket_short":             int(24  < claim.policy_tenure_months <= 60),
        "tenure_bucket_mid":               int(60  < claim.policy_tenure_months <= 120),
        "tenure_bucket_long":              int(claim.policy_tenure_months > 120),
    }
    return np.array([row[f] for f in FEATURES]).reshape(1, -1)


def risk_label(prob: float) -> dict:
    if prob >= 0.75:
        return {"risk_level": "HIGH",     "message": "Very likely fraud — flag for investigation"}
    elif prob >= 0.50:
        return {"risk_level": "MEDIUM",   "message": "Possible fraud — manual review recommended"}
    elif prob >= 0.30:
        return {"risk_level": "LOW",      "message": "Low fraud risk — monitor"}
    else:
        return {"risk_level": "VERY LOW", "message": "Very unlikely fraud — approve"}


def build_result(claim: ClaimInput) -> dict:
    vec  = encode(claim)
    prob = float(MODEL.predict_proba(vec)[0][1])
    pred = prob >= 0.5
    return {"fraud_predicted": pred, "fraud_probability": round(prob, 4), **risk_label(prob)}


@app.get("/", tags=["Info"])
def root():
    return {"name": "Insurance Fraud Detection API", "version": "2.0.0",
            "model": METADATA["model_name"], "roc_auc": METADATA["roc_auc"], "docs": "/docs"}


@app.get("/health", tags=["Info"])
def health():
    return {"status": "healthy", "model": METADATA["model_name"],
            "features": METADATA["n_features"], "accuracy": METADATA["accuracy"],
            "roc_auc": METADATA["roc_auc"], "saved_at": METADATA["saved_at"],
            "timestamp": datetime.utcnow().isoformat()}


@app.get("/features", tags=["Info"])
def get_features():
    return {"total": len(FEATURES), "features": FEATURES}


@app.get("/model/info", tags=["Model"])
def model_info():
    return METADATA


@app.post("/predict", tags=["Prediction"])
def predict(claim: ClaimInput):
    """Predict fraud for a single insurance claim."""
    try:
        result = build_result(claim)
        return {"claim_id": f"CLM-{int(time.time()*1000) % 99999:05d}",
                "prediction": result, "processed_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(payload: BatchInput):
    """Predict fraud for multiple claims (max 100 per request)."""
    try:
        predictions = []
        fraud_count = 0
        for i, claim in enumerate(payload.claims):
            result = build_result(claim)
            if result["fraud_predicted"]:
                fraud_count += 1
            predictions.append({"index": i + 1, **result})
        return {"total": len(predictions), "fraud_count": fraud_count,
                "non_fraud_count": len(predictions) - fraud_count,
                "fraud_rate": round(fraud_count / len(predictions) * 100, 1),
                "predictions": predictions, "processed_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(claim: ClaimInput):
    """Predict fraud + return top 5 features driving the decision."""
    try:
        vec  = encode(claim)
        prob = float(MODEL.predict_proba(vec)[0][1])
        pred = prob >= 0.5
        risk = risk_label(prob)
        contributions = sorted(
            [{"feature": name, "value": round(float(vec[0][i]), 4),
              "importance": round(float(MODEL.feature_importances_[i]), 4)}
             for i, name in enumerate(FEATURES)],
            key=lambda x: x["importance"], reverse=True)
        return {"fraud_predicted": pred, "fraud_probability": round(prob, 4), **risk,
                "top_5_features": contributions[:5], "all_features": contributions,
                "processed_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
