import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Mental Health Text Classifier",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# LOAD MODEL, TOKENIZER AND LABEL ENCODER
# =========================================================

@st.cache_resource
def load_files():

    # Load trained FastText + GRU model
    model = tf.keras.models.load_model(
        "FastText_GRU_best.keras"
    )

    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_files()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # 3. Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # 4. Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # 5. Convert text into sequence
    sequence = tokenizer.texts_to_sequences([text])

    # 6. Padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    return padded_sequence


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_mental_health(text):

    # Preprocess input
    padded_sequence = preprocess_text(text)

    # Model prediction
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Get class with highest probability
    predicted_class = np.argmax(prediction)

    # Convert numerical class back to original label
    predicted_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # Confidence
    confidence = prediction[predicted_class]

    return predicted_label, confidence, prediction


# =========================================================
# STREAMLIT USER INTERFACE
# =========================================================

st.title("🧠 Mental Health Text Classification")

st.write(
    "Enter a text message below and the "
    "FastText + GRU model will classify it."
)

st.info(
    "Possible classes: negative, neutral, positive, very negative"
)


# =========================================================
# TEXT INPUT
# =========================================================

text_input = st.text_area(
    "Enter your text:",
    height=160,
    placeholder="Example: I've been feeling stressed lately because of my exams..."
)


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(
    "🔍 Predict",
    type="primary",
    use_container_width=True
):

    # Check empty input
    if not text_input.strip():

        st.warning(
            "⚠️ Please enter some text first."
        )

    else:

        # Get prediction
        predicted_label, confidence, probabilities = (
            predict_mental_health(text_input)
        )

        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.subheader("Prediction")

        st.success(
            f"Predicted Class: {predicted_label}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # =================================================
        # DISPLAY PROBABILITIES
        # =================================================

        st.subheader("Class Probabilities")

        for label, probability in zip(
            label_encoder.classes_,
            probabilities
        ):

            percentage = probability * 100

            st.write(
                f"**{label}: {percentage:.2f}%**"
            )

            st.progress(
                float(probability)
            )

        # =================================================
        # DISCLAIMER
        # =================================================

        st.caption(
            "Note: This application provides a machine-learning "
            "classification result and is not a medical diagnosis."
        )