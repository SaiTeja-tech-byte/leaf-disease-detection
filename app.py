import streamlit as st
import numpy as np
import json
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Leaf Disease Detection",
    layout="centered"
)

st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image to predict the **crop** and **disease**")

# ---------------- GITHUB RELEASE URLS ----------------
MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/leaf_disease_multicrop_model.keras"
MODEL_PATH = "leaf_disease_multicrop_model.keras"

METRICS_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/model_metrics.3.json"
METRICS_PATH = "model_metrics.json"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

# ---------------- LOAD METRICS ----------------
@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        r = requests.get(METRICS_URL)
        r.raise_for_status()
        with open(METRICS_PATH, "wb") as f:
            f.write(r.content)
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

model = load_trained_model()
metrics = load_metrics()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Decode class label
    class_label = model.class_names[idx]
    crop, disease = class_label.split("__")

    # ---------------- OUTPUT ----------------
    st.success("🧠 Prediction Result")

    st.write(f"🌱 **Crop:** {crop}")
    st.write(f"🦠 **Disease:** {disease.replace('_', ' ')}")
    st.write(f"📊 **Confidence:** {confidence:.2f}%")

    st.progress(confidence / 100)

    st.divider()

    st.info("📈 Model Performance (Test Set)")
    st.write(f"✅ **Accuracy:** {metrics['accuracy'] * 100:.2f}%")
    st.write(f"🎯 **Precision:** {metrics['precision'] * 100:.2f}%")
