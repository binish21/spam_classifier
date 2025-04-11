
from flask import Flask, request, jsonify, render_template
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and tokenizer
model = load_model("sms_spam_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_message(msg):
    msg = re.sub(r'[^\w\s]', '', msg.lower())
    seq = tokenizer.texts_to_sequences([msg])
    padded = pad_sequences(seq, maxlen=100)
    return padded

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    msg = data["message"]
    processed = preprocess_message(msg)
    pred = model.predict(processed).item()
    label = "SPAM" if pred >= 0.5 else "NOT SPAM"
    confidence = round(pred if pred >= 0.5 else 1 - pred, 2)

    return jsonify({
        "message": msg,
        "prediction": label,
        "confidence": f"{confidence * 100:.1f}%"
    })

@app.route("/predict_form", methods=["POST"])
def predict_form():
    msg = request.form["message"]
    processed = preprocess_message(msg)
    pred = model.predict(processed).item()
    label = "SPAM" if pred >= 0.5 else "NOT SPAM"
    confidence = round(pred if pred >= 0.5 else 1 - pred, 2)
    return render_template("index.html", prediction=label, confidence=f"{confidence * 100:.1f}%")

if __name__ == "__main__":
    app.run(debug=True)
