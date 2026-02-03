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

# ================== BACKGROUND & STYLE ==================
st.markdown(
    """
    <style>
    /* Background: blue sky to green grass */
    .stApp {
        background: linear-gradient(
            to bottom,
            #87CEEB 0%,
            #B0E0E6 30%,
            #98FB98 65%,
            #2E8B57 100%
        );
        background-attachment: fixed;
    }

    /* Sun glow */
    .sun {
        position: fixed;
        top: 40px;
        right: 60px;
        width: 140px;
        height: 140px;
        background: radial-gradient(circle, #FFD700 40%, #FFA500 70%, transparent 75%);
        border-radius: 50%;
        box-shadow: 0 0 80px 30px rgba(255, 215, 0, 0.6);
        z-index: -1;
    }

    /* White glass card */
    .block-container {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 18px;
        padding: 2rem;
    }

    h1, h2, h3 {
        color: #1B5E20;
    }

    p, label {
        color: #1f2937;
        font-weight: 500;
    }
    </style>

    <div class="sun"></div>
    """,
    unsafe_allow_html=True
)

# ================== SIDEBAR ==================
st.sidebar.title("🌱 Project Overview")
st.sidebar.markdown("""
**AI-Driven Web Application for  
Automated Crop Disease Detection**

• Deep Learning (MobileNet CNN)  
• Trained on plant leaf images  
• Deployed using Streamlit Cloud  
""")
st.sidebar.markdown("---")
st.sidebar.write("📌 Upload a clear leaf image for best results")

# ================== HEADER ==================
st.markdown(
    "<h1 style='text-align:center;'>🌿 AI-Driven Crop Disease Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Identify the crop and disease from a leaf image</p>",
    unsafe_allow_html=True
)

# ================== MODEL CONFIG ==================
MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v1.0.0/leaf_disease_model_final.keras"
MODEL_PATH = "leaf_disease_model_final.keras"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading trained model..."):
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

    # Dynamic input size from model
    input_height = model.input_shape[1]
    input_width = model.input_shape[2]

    img = image.load_img(uploaded_file, target_size=(input_height, input_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing leaf image..."):
        prediction = model.predict(img_array)

    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    leaf_name, disease_name = disease_info.get(
        predicted_class, ("Unknown Leaf", "Unknown Disease")
    )

    # ================== RESULTS ==================
    st.markdown("---")
    st.markdown("## 🧾 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🌱 **Leaf Identified**\n\n{leaf_name}")

    with col2:
        st.error(f"🦠 **Disease Detected**\n\n{disease_name}")

    st.markdown("### 📊 Prediction Confidence")
    st.progress(int(confidence))
    st.write(f"**{confidence:.2f}% confidence**")

    if confidence < 50:
        st.warning(
            "⚠️ Low confidence prediction. "
            "Please upload a clearer leaf image with good lighting."
        )

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:13px;'>"
    "🌿 Built with Deep Learning & Streamlit | Academic Project"
    "</p>",
    unsafe_allow_html=True
)
