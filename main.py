import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- 1. LOAD MODEL ---
try:
    ort_session = ort.InferenceSession("transaction_fraud_model.onnx")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- 2. IN-MEMORY STATE (Replaces Redis) ---
user_state = {}

# --- 3. INPUT SCHEMA ---
class Transaction(BaseModel):
    user_id: str
    amount: float
    location_id: int
    hour_of_day: int

@app.post("/predict")
def predict_fraud(tx: Transaction):
    # --- A. FEATURE ENGINEERING (The Logic) ---

    # Get current history
    history = user_state.get(tx.user_id, {"sum": 0.0, "count": 0})

    # Calculate features for this specific transaction
    if history["count"] > 0:
        user_avg_amt = history["sum"] / history["count"]
    else:
        user_avg_amt = tx.amount # First time user

    # Avoid division by zero
    if user_avg_amt == 0: user_avg_amt = 1.0

    amt_deviation = tx.amount / user_avg_amt

    # --- B. PREPARE VECTOR FOR ONNX ---
    input_vector = np.array([[
        tx.amount,
        tx.location_id,
        tx.hour_of_day,
        user_avg_amt,
        amt_deviation
    ]], dtype=np.float32)

    # --- C. INFERENCE ---
    inputs = {ort_session.get_inputs()[0].name: input_vector}
    # XGBoost ONNX output: [label, probabilities]
    logits = ort_session.run(None, inputs)

    # logits[1] is a list of maps, e.g., [{0: 0.9, 1: 0.1}]
    probs = logits[1][0]
    fraud_prob = probs[1] # Probability of Class 1 (Fraud)

    # --- D. UPDATE STATE ---
    user_state[tx.user_id] = {
        "sum": history["sum"] + tx.amount,
        "count": history["count"] + 1
    }

    # --- E. DECISION ---
    status = "LEGIT"
    if fraud_prob > 0.90: status = "BLOCK"
    elif fraud_prob > 0.50: status = "REVIEW"

    return {
        "user_id": tx.user_id,
        "amount": tx.amount,
        "fraud_probability": float(fraud_prob),
        "status": status,
        "context": {
            "avg_spend": float(user_avg_amt),
            "deviation": float(amt_deviation)
        }
    }
