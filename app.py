import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI-Driven Web Application for Automated Disease Detection in Rice and Pulse Crops", layout="centered")

st.title("🌿 AI-Driven Web Application for Automated Disease Detection in Rice and Pulse Crops")
st.write("Upload a leaf image to identify the crop and disease")

# ---------------- MODEL CONFIG ----------------
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

# ---------------- CLASS LABELS ----------------
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

# ---------------- LEAF + DISEASE INFO ----------------
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

# ---------------- MODEL ACCURACY ----------------
MODEL_ACCURACY = 97.9  # from test dataset evaluation

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Get input size dynamically from model
    input_height = model.input_shape[1]
    input_width = model.input_shape[2]

    img = image.load_img(
        uploaded_file,
        target_size=(input_height, input_width)
    )

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    leaf_name, disease_name = disease_info.get(
        predicted_class, ("Unknown Leaf", "Unknown Disease")
    )

    # ---------------- OUTPUT ----------------
    st.success(f"🌱 Leaf Identified: **{leaf_name}**")
    st.warning(f"🦠 Disease Detected: **{disease_name}**")
    st.info(f"📊 Prediction Confidence: **{confidence:.2f}%**")
    st.write(f"✅ Model Accuracy (Test Dataset): **{MODEL_ACCURACY}%**")

    if confidence < 50:
        st.warning("⚠️ Low confidence prediction. Please upload a clearer leaf image.")


