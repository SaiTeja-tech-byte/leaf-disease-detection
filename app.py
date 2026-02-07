import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Leaf Disease Detection", layout="centered")

st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image to predict the disease")

MODEL_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v1.0.0/leaf_disease_mobilenet_model.keras"
MODEL_PATH = "leaf_disease_model_final.keras"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

model = load_trained_model()

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

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🦠 Predicted Disease: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
