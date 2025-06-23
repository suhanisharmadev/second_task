import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

import os

if not os.path.exists("Plant_disease_detection_model_pwp.keras"):
    import gdown
    url = "https://drive.google.com/uc?id=1UJRAq4YEmGClu0_mBm-H2UAzLPHt6cHF"
    gdown.download(url, "Plant_disease_detection_model_pwp.keras", quiet=False)

# Set page config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main { background-color: #f6fff7; }
        h1, h2, h3 {
            color: #2e7d32;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a plant leaf image to detect the disease and get details like **cause** and **cure** using a deep learning model.")

# Load the trained model
model = tf.keras.models.load_model("Plant_disease_detection_model_pwp.keras")

# Load disease info JSON
with open("plant_disease.json", "r") as f:
    disease_data = json.load(f)

# Extract class names
class_names = [d["name"] for d in disease_data]

# File uploader
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # Disease info
    disease_info = disease_data[predicted_index]

    # Results
    st.success(f"🩺 **Predicted Disease:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")
    st.write(f"💡 **Cause:** {disease_info['cause']}")
    st.write(f"🧪 **Cure:** {disease_info['cure']}")
