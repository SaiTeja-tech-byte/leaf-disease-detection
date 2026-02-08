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
st.write("Upload a leaf image to predict the **crop and disease**")

IMG_SIZE = (224, 224)

# --------------------------------------------------
# GITHUB RELEASE URLS (CONFIRMED)
# --------------------------------------------------

# v2 model (Rice, Potato, Tomato, Pepper)
MODEL_V2_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/leaf_disease_multicrop_model.keras"
METRICS_V2_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/model_metrics.3.json"

# v3 model (All remaining crops)
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
# HELPER FUNCTIONS
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

V2_CROPS = {"Pepper", "Potato", "Rice", "Tomato"}

# --------------------------------------------------
# LOAD MODELS & METRICS
# --------------------------------------------------
model_v2, model_v3 = load_models()
metrics_v2, metrics_v3 = load_metrics()

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- v2 prediction --------
    preds_v2 = model_v2.predict(img_array)
    idx_v2 = int(np.argmax(preds_v2))
    label_v2 = V2_CLASSES[idx_v2]
    crop_guess = label_v2.split("__")[0]

    # -------- Decide model --------
    if crop_guess in V2_CROPS:
        final_label = label_v2
        confidence = float(np.max(preds_v2)) * 100
        metrics = metrics_v2
        model_used = "v2"
    else:
        preds_v3 = model_v3.predict(img_array)
        idx_v3 = int(np.argmax(preds_v3))
        final_label = V3_CLASSES[idx_v3]
        confidence = float(np.max(preds_v3)) * 100
        metrics = metrics_v3
        model_used = "v3"

    # -------- SAFE crop/disease split (FIXED BUG) --------
    parts = final_label.split("___")
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else "Healthy / No disease detected"

    crop = crop.replace("_", " ").title()
    disease = disease.replace("_", " ").title()

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------
    st.subheader("🧠 Prediction Result")
    st.success(f"🌱 Crop: **{crop}**")
    st.info(f"🦠 Disease: **{disease}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
    st.write(f"🧠 Model used: **{model_used.upper()}**")

    st.progress(float(confidence) / 100.0)

    st.subheader("📈 Model Performance")
    st.write(f"✅ Accuracy: **{metrics['accuracy'] * 100:.2f}%**")
    st.write(f"🎯 Precision: **{metrics['precision'] * 100:.2f}%**")
