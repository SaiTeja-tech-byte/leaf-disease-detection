import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI-Based Crop Leaf Classification",
    page_icon="🌿",
    layout="centered"
)

# ================= BACKGROUND IMAGE =================
BACKGROUND_IMAGE_URL = (
    "https://raw.githubusercontent.com/"
    "SaiTeja-tech-byte/leaf-disease-detection/main/assets/background.jpg"
)

# ================= STYLES =================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{BACKGROUND_IMAGE_URL}") no-repeat center center fixed;
        background-size: cover;
    }}

    .block-container {{
        background: rgba(15,15,15,0.88);
        padding: 2.5rem;
        border-radius: 16px;
        max-width: 900px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
        backdrop-filter: blur(10px);
    }}

    h1, h2, h3 {{
        color: #e5fbe5;
        text-align: center;
    }}

    p, label, span, div {{
        color: #e5e7eb;
    }}

    div[data-testid="stFileUploader"] {{
        background: rgba(30,30,30,0.95);
        padding: 1rem;
        border-radius: 12px;
    }}

    [data-testid="stMetricValue"] {{
        color: #a7f3d0;
    }}

    .footer {{
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ================= MODEL METRICS =================
# (These are crop classification metrics, NOT disease accuracy)
MODEL_ACCURACY = 97.9
MODEL_PRECISION = 92.1

# ================= MODEL CONFIG =================
MODEL_URL = (
    "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/"
    "releases/download/v1.0.0/leaf_disease_model_final.keras"
)
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

# ================= CLASS LABELS (CROP ONLY) =================
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

# ================= HEADER =================
st.markdown("<h1>🌿 AI-Based Crop Leaf Classification</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Upload a leaf image to identify the crop type</p>",
    unsafe_allow_html=True
)

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # Match model input size
    h, w = model.input_shape[1], model.input_shape[2]
    img = image.load_img(uploaded_file, target_size=(h, w))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_crop = class_labels[predicted_index]
    confidence = np.max(prediction) * 100

    # ================= OUTPUT =================
    st.markdown("---")
    st.markdown("## 🧾 Prediction Result")

    st.success(f"🌱 **Predicted Crop:** {predicted_crop}")

    st.markdown("### 📊 Prediction Confidence")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}%")

    st.markdown("### 📈 Model Performance (Crop Classification)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{MODEL_ACCURACY}%")
    c2.metric("Precision", f"{MODEL_PRECISION}%")

    # ================= TOP-3 PREDICTIONS =================
    st.markdown("### 🔍 Top-3 Predictions")
    top3 = np.argsort(prediction[0])[-3:][::-1]
    for i in top3:
        st.write(f"- {class_labels[i]} : {prediction[0][i]*100:.2f}%")

# ================= FOOTER =================
st.markdown(
    "<div class='footer'>AI + Deep Learning | Crop Leaf Classification Project</div>",
    unsafe_allow_html=True
)
