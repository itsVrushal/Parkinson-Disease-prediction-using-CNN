import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Set the page layout to wide to utilize full screen width
st.set_page_config(layout="wide")

# Load all models
spiral_model = load_model('Models/parkinson_disease_detection_model(93%).h5')
mri_model = load_model('Models/parkinson_disease_detection_model(MRI).h5')
wave_model = load_model('Models/parkinson_disease_detection_model(wave).h5')  # Assuming you have a wave model

# Streamlit app title and description
st.title("Parkinson's Disease Detection - Multi-Model")
st.write("""
         Upload three images, and the models will predict whether each image indicates signs of Parkinson's Disease or a Healthy condition.
         """)

# Create three columns with equal width (33%)
col1, col2, col3 = st.columns([1, 1, 1])  # Each column takes up 33% of the width

with col1:
    uploaded_spiral = st.file_uploader("Upload an image for Spiral Model (128x128)", type=["png", "jpg", "jpeg"], key="spiral")
with col2:
    uploaded_mri = st.file_uploader("Upload an image for MRI Model (128x128)", type=["png", "jpg", "jpeg"], key="mri")
with col3:
    uploaded_wave = st.file_uploader("Upload an image for Wave Model (128x128)", type=["png", "jpg", "jpeg"], key="wave")

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    img = image.resize((128, 128))  # Resize to the same size used for training
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict_image(model, img_array):
    """Make a prediction and return confidence score."""
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    return confidence

# Prediction function for each uploaded image
def display_prediction(uploaded_file, model, model_name, col):
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        with col:
            st.image(image, caption=f'Uploaded Image for {model_name}', use_column_width=True)

        # Convert image to RGB and preprocess
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = preprocess_image(image)

        # Predict
        with st.spinner(f'Analyzing image for {model_name}...'):
            confidence = predict_image(model, img_array)

        # Display prediction confidence score in the corresponding column
        with col:
            if confidence > 0.5:
                st.markdown(f"<h3 style='color: red;'>{model_name} predicts Parkinson's Disease with {confidence * 100:.2f}% confidence.</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: green;'>{model_name} predicts Healthy with {(1 - confidence) * 100:.2f}% confidence.</h3>", unsafe_allow_html=True)

# Display predictions for each model side by side
if uploaded_spiral:
    display_prediction(uploaded_spiral, spiral_model, "Spiral Model", col1)
if uploaded_mri:
    display_prediction(uploaded_mri, mri_model, "MRI Model", col2)
if uploaded_wave:
    display_prediction(uploaded_wave, wave_model, "Wave Model", col3)
