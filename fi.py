import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("resnet_brain_tumor.h5")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Image Classifier")
st.write("Upload an MRI image to predict the tumor type")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display image
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0]
    label = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    st.write(f"### 🩺 Predicted Tumor Type: **{label}**")
    st.write(f"Confidence: {confidence*100:.2f}%")
