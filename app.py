import streamlit as st
import numpy as np
import os
import json
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Leaf Disease Detection",
    layout="centered"
)

st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image to predict the disease")

# --------------------------------------------------
# Model & Metrics URLs
# --------------------------------------------------
MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v1.0.0/leaf_disease_mobilenet_model.keras"
MODEL_PATH = "leaf_disease_model.keras"

METRICS_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v1.0.0/model_metrics.json"
METRICS_PATH = "model_metrics.json"

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

# --------------------------------------------------
# Load Metrics
# --------------------------------------------------
@st.cache_resource
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        with st.spinner("📥 Loading model metrics..."):
            r = requests.get(METRICS_URL)
            r.raise_for_status()
            with open(METRICS_PATH, "wb") as f:
                f.write(r.content)
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

model = load_trained_model()
metrics = load_metrics()

MODEL_ACCURACY = metrics["accuracy"]
MODEL_PRECISION = metrics["precision"]

# --------------------------------------------------
# Class Labels (13 classes)
# --------------------------------------------------
class_labels = [
    "Cassava",
    "Rice",
    "Apple",
    "Cherry (including sour)",
    "Corn (Maize)",
    "Grape",
    "Orange",
    "Peach",
    "Bell Pepper",
    "Potato",
    "Squash",
    "Strawberry",
    "Tomato"
]

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")

    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(prediction))

    st.success(f"🦠 **Predicted Disease:** {predicted_class}")

    # --------------------------------------------------
    # Metrics Display
    # --------------------------------------------------
    st.subheader("📊 Model Performance")

    st.write("**Confidence (Image-level)**")
    st.progress(confidence)
    st.write(f"{confidence * 100:.2f}%")

    st.write("**Accuracy (Test Dataset)**")
    st.progress(MODEL_ACCURACY)
    st.write(f"{MODEL_ACCURACY * 100:.2f}%")

    st.write("**Precision (Test Dataset)**")
    st.progress(MODEL_PRECISION)
    st.write(f"{MODEL_PRECISION * 100:.2f}%")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "📌 Accuracy & Precision are evaluated on a held-out test set of 5,741 images. "
    "Confidence is image-specific."
)
