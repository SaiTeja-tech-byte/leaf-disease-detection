import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Leaf Disease Detection", layout="centered")

st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image to predict the disease")

@st.cache_resource
def load_trained_model():
   return load_model("leaf_disease_mobilenet_model.keras")

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

uploaded_file = st.file_uploader(
    "Choose a leaf image", type=["jpg", "png", "jpeg"]
)

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
