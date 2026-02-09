import streamlit as st
import numpy as np
import json
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ===================== AUTH IMPORTS =====================
from auth import create_user_table, signup_user, login_user

# ===================== AUTH SETUP =====================
create_user_table()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ===================== LOGIN / SIGNUP UI =====================
def login_page():
    st.title("🔐 User Authentication")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        st.subheader("Sign Up")
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            if signup_user(new_user, new_pass):
                st.success("✅ Account created. Please login.")
            else:
                st.error("❌ Username already exists")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ===================== PROTECT APP =====================
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ===================== MAIN APP =====================
st.set_page_config(page_title="Leaf Disease Detection", layout="centered")

col1, col2 = st.columns([8, 2])
with col2:
    if st.button("🚪 Logout"):
        logout()

st.title("🌿 Leaf Disease Detection")
st.write("Choose the correct tab based on the crop type, then upload a leaf image.")

IMG_SIZE = (224, 224)

# ===================== URLs =====================
MODEL_V2_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/leaf_disease_multicrop_model.keras"
METRICS_V2_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v2.0.0/model_metrics.3.json"

MODEL_V3_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v3.0.0/leaf_disease_v3_checkpoint.keras"
METRICS_V3_URL = "https://github.com/SaiTeja-tech-byte/leaf-disease-detection/releases/download/v3.0.0/model_metrics_v3.json"

MODEL_V2_PATH = "model_v2.keras"
MODEL_V3_PATH = "model_v3.keras"
METRICS_V2_PATH = "metrics_v2.json"
METRICS_V3_PATH = "metrics_v3.json"

# ===================== HELPERS =====================
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

# ===================== CLASSES =====================
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

model_v2, model_v3 = load_models()
metrics_v2, metrics_v3 = load_metrics()

def parse_label(label):
    label = label.replace("___", "__")
    parts = label.split("__")
    crop = parts[0].replace("_", " ").title()

    if len(parts) > 1 and "healthy" not in parts[1].lower():
        disease = parts[1].replace("_", " ").title()
    else:
        disease = "Healthy"

    return crop, disease

tab1, tab2 = st.tabs(["🌾 v2 Model", "🍎 v3 Model"])

# ===================== V2 TAB =====================
with tab1:
    file = st.file_uploader("Upload leaf image (v2)", type=["jpg", "jpeg", "png"])
    if file:
        st.image(file, width=300)
        img = image.load_img(file, target_size=IMG_SIZE)
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model_v2.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        crop, disease = parse_label(V2_CLASSES[idx])

        st.success(f"🌱 Leaf Name: **{crop}**")
        st.info(f"🦠 Leaf Disease: **{disease}**")
        st.metric("📊 Confidence", f"{confidence:.2f}%")
        st.metric("✅ Accuracy", f"{metrics_v2['accuracy']*100:.2f}%")
        st.metric("🎯 Precision", f"{metrics_v2['precision']*100:.2f}%")

# ===================== V3 TAB =====================
with tab2:
    file = st.file_uploader("Upload leaf image (v3)", type=["jpg", "jpeg", "png"])
    if file:
        st.image(file, width=300)
        img = image.load_img(file, target_size=IMG_SIZE)
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model_v3.predict(img)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        crop, disease = parse_label(V3_CLASSES[idx])

        st.success(f"🌱 Leaf Name: **{crop}**")
        st.info(f"🦠 Leaf Disease: **{disease}**")
        st.metric("📊 Confidence", f"{confidence:.2f}%")
        st.metric("✅ Accuracy", f"{metrics_v3['accuracy']*100:.2f}%")
        st.metric("🎯 Precision", f"{metrics_v3['precision']*100:.2f}%")
