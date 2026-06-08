from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pickle

from tensorflow.keras.models import load_model


# ======================
# CREATE FLASK APP
# ======================

app = Flask(__name__)

CORS(app)


# ======================
# LOAD MODEL
# ======================

model = load_model("resume_ann_model.keras")


# ======================
# LOAD TF-IDF
# ======================

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


# ======================
# LOAD LABEL ENCODER
# ======================

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# ======================
# CLEAN TEXT FUNCTION
# ======================

def clean_text(text):

    import re

    text = str(text).lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


# ======================
# HOME ROUTE
# ======================

@app.route("/")
def home():

    return jsonify({
        "message": "ANN Resume Backend Running"
    })


# ======================
# PREDICTION ROUTE
# ======================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        resume = data["resume"]

        cleaned_resume = clean_text(resume)

        vector = tfidf.transform(
            [cleaned_resume]
        ).toarray()

        prediction = model.predict(vector)

        predicted_class = np.argmax(prediction)

        category = encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = float(
            np.max(prediction)
        ) * 100

        return jsonify({

            "predicted_category": category,
            "confidence": round(confidence, 2)

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# ======================
# RUN APP
# ======================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )