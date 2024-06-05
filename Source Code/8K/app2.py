import os
import streamlit as st
import pickle
import numpy as np
from tqdm.notebook import tqdm





BASE_DIR = 'CustomImages'
WORKING_DIR = ''

directory = BASE_DIR


from PIL import Image

# Streamlit App
st.title("Deep Learning Image Captioning App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image Saved Successfully", use_column_width=True)
    img_path = os.path.join(directory, uploaded_file.name)

    # Save the uploaded image
    image.save(img_path)
   