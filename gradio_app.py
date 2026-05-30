import gradio as gr
import requests
import json

# ─────────────────────────────────────────────
# CONFIG — change port if needed
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000"


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def call_api(endpoint: str, payload: dict) -> dict:
    try:
        res = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to FastAPI. Make sure it is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# TAB 1 — SINGLE PREDICTION
# ─────────────────────────────────────────────
def predict_single(
    age, annual_premium, policy_tenure_months, claim_amount,
    num_claims_last_3yrs, vehicle_age, incident_severity,
    incident_type, police_report, witness_present,
    customer_state, channel, claim_status
):
    payload = {
        "age":                  int(age),
        "annual_premium":       float(annual_premium),
        "policy_tenure_months": int(policy_tenure_months),
        "claim_amount":         float(claim_amount),
        "num_claims_last_3yrs": int(num_claims_last_3yrs),
        "vehicle_age":          float(vehicle_age),
        "incident_severity":    incident_severity,
        "incident_type":        incident_type,
        "police_report":        police_report,
        "witness_present":      witness_present,
        "customer_state":       customer_state,
        "channel":              channel,
        "claim_status":         claim_status,
    }

    result = call_api("/predict", payload)

    if "error" in result:
        return result["error"], "", "", ""

    p = result["prediction"]
    fraud      = "🚨 FRAUD DETECTED" if p["fraud_predicted"] else "✅ NOT FRAUD"
    prob       = f"{p['fraud_probability'] * 100:.1f}%"
    risk       = p["risk_level"]
    message    = p["message"]
    claim_id   = result["claim_id"]
    processed  = result["processed_at"]

    summary = f"""
╔══════════════════════════════════════════╗
  Claim ID   : {claim_id}
  Verdict    : {fraud}
  Probability: {prob}
  Risk Level : {risk}
  Message    : {message}
  Processed  : {processed}
╚══════════════════════════════════════════╝
""".strip()

    return summary, fraud, prob, risk


# ─────────────────────────────────────────────
# TAB 2 — PREDICT + EXPLAIN
# ─────────────────────────────────────────────
def predict_explain(
    age, annual_premium, policy_tenure_months, claim_amount,
    num_claims_last_3yrs, vehicle_age, incident_severity,
    incident_type, police_report, witness_present,
    customer_state, channel, claim_status
):
    payload = {
        "age":                  int(age),
        "annual_premium":       float(annual_premium),
        "policy_tenure_months": int(policy_tenure_months),
        "claim_amount":         float(claim_amount),
        "num_claims_last_3yrs": int(num_claims_last_3yrs),
        "vehicle_age":          float(vehicle_age),
        "incident_severity":    incident_severity,
        "incident_type":        incident_type,
        "police_report":        police_report,
        "witness_present":      witness_present,
        "customer_state":       customer_state,
        "channel":              channel,
        "claim_status":         claim_status,
    }

    result = call_api("/predict/explain", payload)

    if "error" in result:
        return result["error"], ""

    fraud   = "🚨 FRAUD DETECTED" if result["fraud_predicted"] else "✅ NOT FRAUD"
    prob    = f"{result['fraud_probability'] * 100:.1f}%"
    risk    = result["risk_level"]
    message = result["message"]

    # Top 5 features table
    top5 = result.get("top_5_features", [])
    feat_lines = ["Rank  Feature                        Value      Importance",
                  "─" * 58]
    for i, f in enumerate(top5, 1):
        feat_lines.append(
            f"  {i}   {f['feature']:<30} {str(f['value']):<10} {f['importance']:.4f}"
        )

    summary = f"""
╔══════════════════════════════════════════╗
  Verdict    : {fraud}
  Probability: {prob}
  Risk Level : {risk}
  Message    : {message}
╚══════════════════════════════════════════╝

TOP 5 FEATURES DRIVING THIS DECISION:
{chr(10).join(feat_lines)}
""".strip()

    return summary, "\n".join(feat_lines)


# ─────────────────────────────────────────────
# TAB 3 — BATCH PREDICTION
# ─────────────────────────────────────────────
def predict_batch_csv(file):
    if file is None:
        return "Please upload a CSV file."
    try:
        import pandas as pd
        df = pd.read_csv(file.name)

        required = [
            "age", "annual_premium", "policy_tenure_months", "claim_amount",
            "num_claims_last_3yrs", "vehicle_age", "incident_severity",
            "incident_type", "police_report", "witness_present",
            "customer_state", "channel", "claim_status"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return f"Missing columns in CSV: {missing}"

        claims = df[required].to_dict(orient="records")
        payload = {"claims": claims}
        result = call_api("/predict/batch", payload)

        if "error" in result:
            return result["error"]

        lines = [
            f"Total Claims  : {result['total']}",
            f"Fraud Count   : {result['fraud_count']}",
            f"Non-Fraud     : {result['non_fraud_count']}",
            f"Fraud Rate    : {result['fraud_rate']}%",
            f"Processed At  : {result['processed_at']}",
            "",
            "─" * 60,
            f"{'#':<5} {'Fraud':<8} {'Probability':<14} {'Risk Level':<12} Message",
            "─" * 60,
        ]
        for p in result["predictions"]:
            flag = "YES 🚨" if p["fraud_predicted"] else "NO  ✅"
            lines.append(
                f"{p['index']:<5} {flag:<8} {str(p['fraud_probability']):<14} "
                f"{p['risk_level']:<12} {p['message']}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {str(e)}"


# ─────────────────────────────────────────────
# TAB 4 — HEALTH CHECK
# ─────────────────────────────────────────────
def check_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        data = res.json()
        return f"""
API Status  : {data.get('status', 'unknown').upper()}
Model       : {data.get('model', '-')}
Accuracy    : {data.get('accuracy', '-')}
ROC-AUC     : {data.get('roc_auc', '-')}
Timestamp   : {data.get('timestamp', '-')}
""".strip()
    except Exception:
        return "❌ FastAPI server is not running. Start it with:\nuvicorn app:app --reload --port 8000"


# ─────────────────────────────────────────────
# SHARED INPUT COMPONENTS
# ─────────────────────────────────────────────
def input_components():
    return [
        gr.Number(label="Age",                  value=35,       minimum=18, maximum=100),
        gr.Number(label="Annual Premium (₹)",   value=25000,    minimum=1),
        gr.Number(label="Policy Tenure (months)",value=36,      minimum=1),
        gr.Number(label="Claim Amount (₹)",     value=180000,   minimum=1),
        gr.Number(label="Claims Last 3 Years",  value=3,        minimum=0, maximum=20),
        gr.Number(label="Vehicle Age (years)",  value=2,        minimum=0),
        gr.Dropdown(label="Incident Severity",  value="Total Loss",
                    choices=["Minor", "Major", "Total Loss"]),
        gr.Dropdown(label="Incident Type",      value="Fire",
                    choices=["Collision", "Fire", "Theft", "Natural Disaster"]),
        gr.Radio(label="Police Report Filed",   value="No",  choices=["Yes", "No"]),
        gr.Radio(label="Witness Present",       value="No",  choices=["Yes", "No"]),
        gr.Dropdown(label="Customer State",     value="KA",
                    choices=["KA", "MH", "AP", "TS", "TN", "DL"]),
        gr.Dropdown(label="Channel",            value="Online",
                    choices=["Agent", "Branch", "Online"]),
        gr.Dropdown(label="Claim Status",       value="Open",
                    choices=["Open", "Closed", "Under Review"]),
    ]


# ─────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────
with gr.Blocks(title="Insurance Fraud Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🛡️ Insurance Fraud Detection
    **Model**: Random Forest &nbsp;|&nbsp; **ROC-AUC**: 0.9618 &nbsp;|&nbsp; **Recall**: 92.4%
    > Make sure FastAPI is running: `uvicorn app:app --reload --port 8000`
    """)

    with gr.Tabs():

        # ── Tab 1: Single Predict ──
        with gr.Tab("🔍 Single Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Claim Details")
                    inputs1 = input_components()
                    btn1 = gr.Button("Predict Fraud", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    out_summary = gr.Textbox(label="Summary",        lines=9,  interactive=False)
                    out_verdict = gr.Textbox(label="Verdict",        lines=1,  interactive=False)
                    out_prob    = gr.Textbox(label="Probability",     lines=1,  interactive=False)
                    out_risk    = gr.Textbox(label="Risk Level",      lines=1,  interactive=False)

            btn1.click(fn=predict_single, inputs=inputs1,
                       outputs=[out_summary, out_verdict, out_prob, out_risk])

        # ── Tab 2: Explain ──
        with gr.Tab("💡 Predict + Explain"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Claim Details")
                    inputs2 = input_components()
                    btn2 = gr.Button("Predict & Explain", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Result + Feature Breakdown")
                    out_explain  = gr.Textbox(label="Full Result",         lines=12, interactive=False)
                    out_features = gr.Textbox(label="Top 5 Features Table", lines=8, interactive=False)

            btn2.click(fn=predict_explain, inputs=inputs2,
                       outputs=[out_explain, out_features])

        # ── Tab 3: Batch CSV ──
        with gr.Tab("📂 Batch Prediction (CSV)"):
            gr.Markdown("""
            ### Upload a CSV file with these columns:
            `age, annual_premium, policy_tenure_months, claim_amount, num_claims_last_3yrs,
            vehicle_age, incident_severity, incident_type, police_report, witness_present,
            customer_state, channel, claim_status`
            """)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            btn3       = gr.Button("Run Batch Predict", variant="primary")
            out_batch  = gr.Textbox(label="Batch Results", lines=20, interactive=False)

            btn3.click(fn=predict_batch_csv, inputs=[file_input], outputs=[out_batch])

        # ── Tab 4: Health ──
        with gr.Tab("❤️ API Health"):
            gr.Markdown("### Check if FastAPI server is running")
            btn4       = gr.Button("Check Health", variant="secondary")
            out_health = gr.Textbox(label="API Status", lines=6, interactive=False)

            btn4.click(fn=check_health, inputs=[], outputs=[out_health])

    gr.Markdown("---\n*Insurance Fraud Detection — Random Forest Model*")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)