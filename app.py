import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AI-Based Crop Disease Detection System")
st.write("Upload a leaf image to identify the crop, disease, and recommended actions.")

# =========================================================
# MODEL CONFIG (GitHub Release)
# =========================================================
MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/plant_disease_model.keras"
MODEL_PATH = "plant_disease_model.keras"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return load_model(MODEL_PATH)

model = load_trained_model()

# =========================================================
# CLASS LABELS (PlantVillage-style: crop__disease)
# IMPORTANT: Must match training order
# =========================================================
class_labels = list(model.class_names) if hasattr(model, "class_names") else None

# If model does not store class_names, define manually
if class_labels is None:
    class_labels = [
        "tomato__late_blight",
        "tomato__healthy",
        "potato__early_blight",
        "potato__late_blight",
        "potato__healthy",
        "rice__brown_spot",
        "rice__leaf_blast",
        "rice__healthy"
    ]

# =========================================================
# REMEDIES & PRECAUTIONS DATABASE
# =========================================================
REMEDIES = {
    "tomato__late_blight": [
        "Apply copper-based fungicide",
        "Remove infected leaves",
        "Improve air circulation"
    ],
    "potato__early_blight": [
        "Use recommended fungicides",
        "Remove infected foliage",
        "Avoid overhead irrigation"
    ],
    "potato__late_blight": [
        "Apply protective fungicides",
        "Destroy infected plants",
        "Avoid wet soil conditions"
    ],
    "rice__brown_spot": [
        "Use balanced fertilization",
        "Apply fungicides if severe",
        "Remove infected debris"
    ],
    "rice__leaf_blast": [
        "Use resistant varieties",
        "Apply systemic fungicide",
        "Avoid excess nitrogen"
    ]
}

PRECAUTIONS = {
    "tomato__late_blight": [
        "Avoid watering leaves",
        "Use disease-resistant varieties",
        "Practice crop rotation"
    ],
    "potato__early_blight": [
        "Use certified seeds",
        "Ensure good drainage",
        "Rotate crops yearly"
    ],
    "potato__late_blight": [
        "Avoid overcrowding plants",
        "Use clean planting material",
        "Remove volunteer plants"
    ],
    "rice__brown_spot": [
        "Avoid nutrient deficiency",
        "Maintain field hygiene",
        "Ensure proper irrigation"
    ],
    "rice__leaf_blast": [
        "Avoid excess nitrogen fertilizer",
        "Ensure proper spacing",
        "Monitor fields regularly"
    ]
}

MODEL_ACCURACY = 89.7  # Use your TEST accuracy here

# =========================================================
# IMAGE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a leaf image (jpg / png / jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # Match model input size
    input_h, input_w = model.input_shape[1], model.input_shape[2]
    img = image.load_img(uploaded_file, target_size=(input_h, input_w))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)

    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)
    predicted_label = class_labels[idx]

    crop, disease = predicted_label.split("__")

    # =====================================================
    # OUTPUT
    # =====================================================
    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    st.success(f"🌱 Crop: **{crop.capitalize()}**")
    st.warning(f"🦠 Disease: **{disease.replace('_',' ').title()}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
    st.write(f"🎯 Model Accuracy (Test): **{MODEL_ACCURACY}%**")

    # =====================================================
    # REMEDIES
    # =====================================================
    st.markdown("### 💊 Recommended Remedies")
    for remedy in REMEDIES.get(predicted_label, ["No remedy information available"]):
        st.write("•", remedy)

    # =====================================================
    # PRECAUTIONS
    # =====================================================
    st.markdown("### 🛡 Preventive Measures")
    for precaution in PRECAUTIONS.get(predicted_label, ["No precaution information available"]):
        st.write("•", precaution)

    # =====================================================
    # TOP-3 PREDICTIONS
    # =====================================================
    st.markdown("### 🔍 Top-3 Predictions")
    top3 = np.argsort(prediction[0])[-3:][::-1]
    for i in top3:
        label = class_labels[i]
        c, d = label.split("__")
        st.write(f"- {c.capitalize()} ({d.replace('_',' ').title()}): {prediction[0][i]*100:.2f}%")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<hr><center>AI + Deep Learning | Crop Disease Detection System</center>",
    unsafe_allow_html=True
)
