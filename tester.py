import requests
import random
import time

URL = "http://localhost:8000/predict"

def generate_and_send():
    user_ids = [f"user_{i}" for i in range(100)] # Smaller pool for testing

    print("🚀 Sending transactions to backend...")

    while True:
        user = random.choice(user_ids)
        is_fraud = random.random() < 0.1 # 10% chance for demo visibility

        if is_fraud:
            amount = round(random.uniform(200, 2000), 2)
            location_id = 3
        else:
            amount = round(random.uniform(10, 100), 2)
            location_id = random.choices([0, 1, 2])[0]

        payload = {
            "user_id": user,
            "amount": amount,
            "location_id": location_id,
            "hour_of_day": random.randint(0, 23)
        }

        try:
            # Send Request
            response = requests.post(URL, json=payload)
            data = response.json()

            # Print Logic
            status = data['status']
            prob = data['fraud_probability']

            if status == "BLOCK":
                print(f"🔴 BLOCKED! ${amount} (Prob: {prob:.2f}) | User Avg: ${data['context']['avg_spend']:.2f}")
            elif status == "REVIEW":
                print(f"🟡 REVIEW   ${amount} (Prob: {prob:.2f})")
            else:
                print(f"🟢 Legit    ${amount} (Prob: {prob:.2f})")

        except Exception as e:
            print(f"Error: {e}")
            break

        time.sleep(0.1)

if __name__ == "__main__":
    generate_and_send()
