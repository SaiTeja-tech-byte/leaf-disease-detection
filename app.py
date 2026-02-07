import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================================================
# STREAMLIT DEBUG (SHOW ERRORS IF ANY)
# =====================================================
st.set_option("client.showErrorDetails", True)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AI-Based Crop Disease Detection System")
st.write("Upload a leaf image to identify the crop, disease, confidence, remedies and precautions.")

# =====================================================
# MODEL CONFIG (GITHUB RELEASE)
# =====================================================
MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/trained_model.1.keras"
MODEL_PATH = "plant_disease_model.keras"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

model = load_trained_model()

# =====================================================
# CLASS LABELS (MUST MATCH TRAINING)
# =====================================================
class_labels = [
    "tomato__healthy",
    "tomato__late_blight",
    "potato__healthy",
    "potato__early_blight",
    "potato__late_blight",
    "rice__healthy",
    "rice__brown_spot",
    "rice__leaf_blast"
]

# =====================================================
# REMEDIES & PRECAUTIONS
# =====================================================
REMEDIES = {
    "tomato__late_blight": [
        "Apply copper-based fungicide",
        "Remove infected leaves immediately",
        "Improve air circulation"
    ],
    "potato__early_blight": [
        "Use recommended fungicide",
        "Remove infected foliage",
        "Avoid overhead irrigation"
    ],
    "potato__late_blight": [
        "Apply protective fungicides",
        "Destroy infected plants",
        "Avoid water stagnation"
    ],
    "rice__brown_spot": [
        "Apply balanced fertilizer",
        "Use fungicide if severe",
        "Remove crop debris"
    ],
    "rice__leaf_blast": [
        "Use resistant rice varieties",
        "Apply systemic fungicide",
        "Avoid excess nitrogen"
    ]
}

PRECAUTIONS = {
    "tomato__late_blight": [
        "Avoid wet leaves",
        "Practice crop rotation",
        "Use disease-resistant seeds"
    ],
    "potato__early_blight": [
        "Use certified seeds",
        "Maintain proper spacing",
        "Ensure good drainage"
    ],
    "potato__late_blight": [
        "Avoid overcrowding plants",
        "Monitor fields regularly",
        "Remove volunteer plants"
    ],
    "rice__brown_spot": [
        "Maintain soil nutrients",
        "Ensure proper irrigation",
        "Keep field clean"
    ],
    "rice__leaf_blast": [
        "Avoid excess fertilizer",
        "Ensure proper spacing",
        "Monitor humidity levels"
    ]
}

# =====================================================
# MODEL ACCURACY (TEST SET)
# =====================================================
MODEL_ACCURACY = 89.7  # replace if your test accuracy changes

# =====================================================
# IMAGE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG / PNG / JPEG)",
    type=["jpg", "png", "jpeg"]
)

# =====================================================
# PREDICTION
# =====================================================
if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

        # Prepare image
        input_h = model.input_shape[1]
        input_w = model.input_shape[2]

        img = image.load_img(uploaded_file, target_size=(input_h, input_w))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        predicted_label = class_labels[predicted_index]

        crop, disease = predicted_label.split("__")

        # =================================================
        # OUTPUT
        # =================================================
        st.markdown("---")
        st.subheader("📌 Prediction Result")

        st.success(f"🌱 Crop Identified: **{crop.capitalize()}**")
        st.error(f"🦠 Disease Detected: **{disease.replace('_',' ').title()}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        st.write(f"🎯 Model Accuracy (Test): **{MODEL_ACCURACY}%**")

        # Remedies
        st.markdown("### 💊 Remedies")
        for r in REMEDIES.get(predicted_label, ["No remedy information available"]):
            st.write("•", r)

        # Precautions
        st.markdown("### 🛡 Precautions")
        for p in PRECAUTIONS.get(predicted_label, ["No precaution information available"]):
            st.write("•", p)

        # Top-3 predictions
        st.markdown("### 🔍 Top-3 Predictions")
        top3 = np.argsort(prediction[0])[-3:][::-1]
        for i in top3:
            lbl = class_labels[i]
            c, d = lbl.split("__")
            st.write(
                f"- {c.capitalize()} ({d.replace('_',' ').title()}): {prediction[0][i]*100:.2f}%"
            )

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<hr><center>AI-Driven Crop Disease Detection | Deep Learning Project</center>",
    unsafe_allow_html=True
)

