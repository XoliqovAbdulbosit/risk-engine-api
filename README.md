# 🛡️ Real-Time Transaction Fraud Detection System

A high-performance machine learning pipeline designed to detect fraudulent credit card transactions in sub-millisecond timeframes. This project demonstrates the integration of **stateful feature engineering**, **MLOps best practices**, and **real-time inference**.

---

## 🏗️ Architecture

The system mimics a production Fintech environment (e.g., Stripe/PayPal) where decision latency is critical.

**The Pipeline:**
1.  **Ingestion:** Accepts transaction streams via REST API (simulating a Payment Gateway).
2.  **State Management:** Maintains user history (Running Sums/Counts) in-memory to calculate real-time behavioral features (e.g., *current_amount vs. user_average*).
3.  **Inference Engine:** Uses **ONNX Runtime** to execute an **XGBoost** model. ONNX was chosen over standard Pickling for cross-platform interoperability and faster C++ based inference.
4.  **Decision Logic:** Implements dynamic thresholding:
    *   `> 0.90`: **BLOCK** (High Confidence Fraud)
    *   `> 0.50`: **REVIEW** (Suspicious)
    *   `< 0.50`: **LEGIT**

---

## 🛠️ Tech Stack

*   **Machine Learning:** XGBoost (Gradient Boosting), Scikit-Learn.
*   **MLOps & Serving:** ONNX (Open Neural Network Exchange), ONNX Runtime.
*   **Backend:** FastAPI (Python), Uvicorn.
*   **Data Processing:** NumPy, Pandas.

---

## 🚀 How to Run

### 1. Setup
Clone the repo and install dependencies:
```bash
# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Start the Inference Server
This starts the FastAPI backend which loads the `.onnx` model into memory.
```bash
uvicorn main:app --reload
```
*Server will listen on `http://localhost:8000`*

### 3. Start the Simulation
In a new terminal, run the tester script. This acts as the "World," generating synthetic transactions and sending them to the API.
```bash
python tester.py
```

---

## 📊 Key Engineering Decisions

### 1. Why ONNX?
In a Python-based ML workflow, models are typically saved as Pickle files. However, Pickle is slow and Python-specific.
*   **Decision:** I exported the trained XGBoost model to **ONNX**.
*   **Benefit:** This decouples the model from the training environment and allows for potential deployment in high-performance environments (C++/Go) or Edge devices, reducing inference latency significantly.

### 2. The "Cold Start" & State Problem
A raw transaction (e.g., "$500 at Target") is rarely enough to detect fraud. We need context: *"Does this user usually spend $500?"*
*   **Implementation:** The backend maintains a stateful history of the user.
*   **Feature Engineering:** On every request, the system calculates `amount_deviation = current_amount / user_average`. This dynamic feature is the strongest predictor of fraud in the system.

### 3. Precision vs. Recall Trade-off
The model was tuned to prioritize **Recall** (catching as much fraud as possible) over Precision.
*   **Business Logic:** In Fintech, a False Negative (missing a $2,000 fraud) is costlier than a False Positive (sending an SMS verification to a legit user).
*   **Solution:** A tiered alert system (Block vs. Review) manages the False Positives without blocking legitimate revenue.

---

## 📈 Future Roadmap

*   **Redis Integration:** Move in-memory state to Redis for persistence and horizontal scaling.
*   **Kafka Streaming:** Decouple Producer and Consumer using Apache Kafka for high-throughput buffering.
*   **Dashboarding:** Build a Next.js frontend with WebSockets to visualize the fraud stream in real-time.
