import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI-Driven Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ================== CLEAN UI STYLE ==================
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #eef7f1; /* soft nature background */
    }

    /* Main container */
    .block-container {
        background-color: #black;
        padding: 2.5rem;
        border-radius: 16px;
        max-width: 850px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }

    /* Title */
    h1 {
        color: #1b5e20;
        text-align: center;
        font-weight: 700;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* Section headers */
    h3 {
        color: #1b5e20;
    }

    /* Remove extra borders feeling */
    div[data-testid="stFileUploader"] {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== HEADER ==================
st.markdown("<h1>🌿 AI-Driven Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload a leaf image to identify the crop and disease</div>",
    unsafe_allow_html=True
)

# ================== MODEL CONFIG ==================
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

# ================== CLASS LABELS ==================
class_labels = [
    "Cassava",
    "Rice",
    "apple",
    "cherry (including sour)",
    "corn (maize)",
    "grape",
    "orange",
    "peach",
    "pepper, bell",
    "potato",
    "squash",
    "strawberry",
    "tomato"
]

# ================== LEAF & DISEASE INFO ==================
disease_info = {
    "Cassava": ("Cassava Leaf", "Cassava Mosaic Disease"),
    "Rice": ("Rice Leaf", "Brown Spot"),
    "apple": ("Apple Leaf", "Apple Scab"),
    "cherry (including sour)": ("Cherry Leaf", "Powdery Mildew"),
    "corn (maize)": ("Corn Leaf", "Leaf Blight"),
    "grape": ("Grape Leaf", "Black Rot"),
    "orange": ("Orange Leaf", "Citrus Canker"),
    "peach": ("Peach Leaf", "Bacterial Spot"),
    "pepper, bell": ("Bell Pepper Leaf", "Bacterial Spot"),
    "potato": ("Potato Leaf", "Early Blight"),
    "squash": ("Squash Leaf", "Powdery Mildew"),
    "strawberry": ("Strawberry Leaf", "Leaf Scorch"),
    "tomato": ("Tomato Leaf", "Late Blight / Leaf Mold")
}

# ================== IMAGE UPLOAD ==================
st.markdown("### 📤 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    input_height = model.input_shape[1]
    input_width = model.input_shape[2]

    img = image.load_img(uploaded_file, target_size=(input_height, input_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing leaf image..."):
        prediction = model.predict(img_array)

    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    leaf_name, disease_name = disease_info.get(
        predicted_class, ("Unknown Leaf", "Unknown Disease")
    )

    st.markdown("---")
    st.markdown("### 🧾 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🌱 Leaf Identified\n\n**{leaf_name}**")

    with col2:
        st.error(f"🦠 Disease Detected\n\n**{disease_name}**")

    st.markdown("### 📊 Prediction Confidence")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}% confidence")

    if confidence < 50:
        st.warning("Low confidence prediction. Please upload a clearer leaf image.")

# ================== FOOTER ==================
st.markdown(
    "<div class='footer'>Built with Deep Learning & Streamlit | Academic Project</div>",
    unsafe_allow_html=True
)

