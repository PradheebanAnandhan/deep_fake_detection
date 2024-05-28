import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deepfake Image Detector", page_icon="🕵️‍♂️", layout="wide")

model_path = r"D:\Python\best_model.h5"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def preprocess_image(image, target_size=(128, 128)):
    try:
        img = image.resize(target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_image(image):
    img_array = preprocess_image(image)
    if img_array is None:
        return None
    try:
        prediction = model.predict(img_array)
        result = 'Fake' if prediction[0] < 0.5 else 'Real'
        return result
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None

# UI for the sidebar
st.sidebar.title("Deepfake Image Detector")
st.sidebar.write("Upload an image to predict if it is fake or real.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    prediction = predict_image(image)
    if prediction:
        st.success(f"**The image is predicted to be: {prediction}**")
    else:
        st.write("Prediction could not be made.")
else:
    st.write("Please upload an image to classify.")

st.markdown(
    """
    <style>
    .stSidebar {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-color: #c3e6cb !important;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)
