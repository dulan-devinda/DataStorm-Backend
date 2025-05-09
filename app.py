import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Optional: enable CORS if frontend runs on a different port

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Paste your actual feature columns here
expected_columns = [...]  # Example: ['feature1', 'feature2', ...]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
        X = pd.get_dummies(df.drop("agent_code", axis=1))
        X = X.reindex(columns=expected_columns, fill_value=0)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
