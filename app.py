import streamlit as st
import numpy as np
import json
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")
st.title("🌿 Leaf Disease Detection")
st.write("Choose the correct tab based on the crop type, then upload a leaf image.")

IMG_SIZE = (224, 224)

# --------------------------------------------------
# GITHUB RELEASE URLS
# --------------------------------------------------

# v2 model (Rice, Potato, Tomato, Pepper)
MODEL_V2_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/leaf_disease_multicrop_model.keras"
METRICS_V2_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/model_metrics.3.json"

# v3 model (Fruits & others)
MODEL_V3_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v3.0.0/leaf_disease_v3_checkpoint.keras"
METRICS_V3_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v3.0.0/model_metrics_v3.json"

# --------------------------------------------------
# LOCAL PATHS
# --------------------------------------------------
MODEL_V2_PATH = "model_v2.keras"
MODEL_V3_PATH = "model_v3.keras"
METRICS_V2_PATH = "metrics_v2.json"
METRICS_V3_PATH = "metrics_v3.json"

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

@st.cache_resource
def load_models():
    download_file(MODEL_V2_URL, MODEL_V2_PATH)
    download_file(MODEL_V3_URL, MODEL_V3_PATH)
    return load_model(MODEL_V2_PATH), load_model(MODEL_V3_PATH)

@st.cache_data
def load_metrics():
    download_file(METRICS_V2_URL, METRICS_V2_PATH)
    download_file(METRICS_V3_URL, METRICS_V3_PATH)
    with open(METRICS_V2_PATH) as f:
        m2 = json.load(f)
    with open(METRICS_V3_PATH) as f:
        m3 = json.load(f)
    return m2, m3

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------
V2_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice__Bacterial_leaf_blight",
    "Rice__Brown_spot",
    "Rice__Leaf_smut",
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__healthy",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_YellowLeaf__Curl_Virus"
]

V3_CLASSES = sorted([
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch"
])

# --------------------------------------------------
# LOAD MODELS & METRICS
# --------------------------------------------------
model_v2, model_v3 = load_models()
metrics_v2, metrics_v3 = load_metrics()

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2 = st.tabs(
    ["🌾 v2 Model (Rice / Potato / Tomato / Pepper)",
     "🍎 v3 Model (Fruits & Other Crops)"]
)

# =========================
# TAB 1 — v2 MODEL
# =========================
with tab1:
    st.subheader("🌾 v2 Model")
    st.write("Supported crops: Rice, Potato, Tomato, Pepper")

    file = st.file_uploader(
        "Upload leaf image (v2)",
        type=["jpg", "jpeg", "png"],
        key="v2_upload"
    )

    if file:
        st.image(file, width=300)

        img = image.load_img(file, target_size=IMG_SIZE)
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model_v2.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        label = V2_CLASSES[idx]
        parts = label.split("___")

        crop = parts[0].replace("_", " ").title()
        disease = (
            parts[1].replace("_", " ").title()
            if len(parts) > 1
            else "Healthy / No disease detected"
        )

        st.success(f"🌱 Leaf Name: **{crop}**")
        st.info(f"🦠 Leaf Disease: **{disease}**")
        st.metric("📊 Confidence", f"{confidence:.2f}%")
        st.metric("✅ Accuracy", f"{metrics_v2['accuracy'] * 100:.2f}%")
        st.metric("🎯 Precision", f"{metrics_v2['precision'] * 100:.2f}%")

# =========================
# TAB 2 — v3 MODEL
# =========================
with tab2:
    st.subheader("🍎 v3 Model")
    st.write("Supported crops: Fruits, Corn, Soybean, Squash")

    file = st.file_uploader(
        "Upload leaf image (v3)",
        type=["jpg", "jpeg", "png"],
        key="v3_upload"
    )

    if file:
        st.image(file, width=300)

        img = image.load_img(file, target_size=IMG_SIZE)
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model_v3.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        label = V3_CLASSES[idx]
        parts = label.split("___")

        crop = parts[0].replace("_", " ").title()
        disease = (
            parts[1].replace("_", " ").title()
            if len(parts) > 1
            else "Healthy / No disease detected"
        )

        st.success(f"🌱 Leaf Name: **{crop}**")
        st.info(f"🦠 Leaf Disease: **{disease}**")
        st.metric("📊 Confidence", f"{confidence:.2f}%")
        st.metric("✅ Accuracy", f"{metrics_v3['accuracy'] * 100:.2f}%")
        st.metric("🎯 Precision", f"{metrics_v3['precision'] * 100:.2f}%")
