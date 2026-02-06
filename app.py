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
    page_title="AI-Based Crop Leaf Classification",
    page_icon="🌿",
    layout="centered"
)

# =========================================================
# CUSTOM STYLES (DARK, CLEAN, READABLE)
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
    }

    .block-container {
        background: rgba(0, 0, 0, 0.75);
        padding: 2.5rem;
        border-radius: 18px;
        max-width: 850px;
        box-shadow: 0px 20px 40px rgba(0,0,0,0.6);
    }

    h1, h2, h3 {
        color: #e6fffa;
        text-align: center;
    }

    p, span, label, div {
        color: #e5e7eb;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(20,20,20,0.9);
        padding: 1rem;
        border-radius: 12px;
    }

    .warning-box {
        background: rgba(255, 193, 7, 0.15);
        padding: 1rem;
        border-radius: 10px;
        border-left: 6px solid #ffc107;
        margin-top: 1rem;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MODEL CONFIG
# =========================================================
MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v1.0.0/leaf_disease_model_final.keras"
MODEL_PATH = "leaf_disease_model_final.keras"

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

# =========================================================
# CLASS LABELS (CROP ONLY — HONEST)
# =========================================================
class_labels = [
    "Cassava",
    "Rice",
    "Apple",
    "Cherry",
    "Corn (Maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Squash",
    "Strawberry",
    "Tomato"
]

# =========================================================
# HEADER
# =========================================================
st.markdown("<h1>🌿 Crop Leaf Classification System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Upload a leaf image to identify the crop type</p>",
    unsafe_allow_html=True
)

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a leaf image (jpg / png / jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # Match model input size dynamically
    input_h, input_w = model.input_shape[1], model.input_shape[2]
    img = image.load_img(uploaded_file, target_size=(input_h, input_w))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_crop = class_labels[predicted_index]
    confidence = float(np.max(prediction) * 100)

    # =====================================================
    # RESULTS
    # =====================================================
    st.markdown("---")
    st.markdown("## 🧾 Prediction Result")

    st.success(f"🌱 **Predicted Crop:** {predicted_crop}")

    st.markdown("### 📊 Prediction Confidence")
    st.progress(int(confidence))
    st.write(f"**{confidence:.2f}%**")

    # =====================================================
    # IMPORTANT WARNING (THIS FIXES YOUR ISSUE)
    # =====================================================
    st.markdown(
        """
        <div class="warning-box">
        ⚠️ <b>Important Note:</b><br>
        This system performs <b>crop leaf classification only</b>.<br>
        It does <b>NOT</b> identify plant diseases.<br><br>
        Disease-infected leaves may affect prediction accuracy.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # TOP-3 PREDICTIONS (TRUST & TRANSPARENCY)
    # =====================================================
    st.markdown("### 🔍 Top-3 Predictions")
    top3 = np.argsort(prediction[0])[-3:][::-1]
    for idx in top3:
        st.write(f"- {class_labels[idx]} : {prediction[0][idx]*100:.2f}%")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<div class='footer'>AI + Deep Learning | Crop Leaf Classification Project</div>",
    unsafe_allow_html=True
)
