import os
import streamlit as st
import pickle
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, Embedding, GRU, Concatenate, Reshape, Dropout, add
import pandas as pd
import re
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers


BASE_DIR = 'Dataset'
WORKING_DIR = ''
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r', encoding='utf-8') as f:
    next(f)
    captions_doc = f.read()
# Create mapping of image to captions
mapping = {}
# Process lines
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)

    if image_id not in mapping:
        mapping[image_id] = []

    mapping[image_id].append(caption)

# Define the clean function
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = re.sub(r'[^a-zA-Z]', ' ', caption)
            caption = re.sub(r'\s+', ' ', caption)
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

# Preprocess the text
clean(mapping)
# Collect all captions
all_captions = [caption for key in mapping for caption in mapping[key]]
# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
# Get the maximum length of the caption
max_length = max(len(caption.split()) for caption in all_captions)
# Get the list of image IDs
image_ids = list(mapping.keys())
BASE_DIR = 'CustomImages'
WORKING_DIR = ''
# Load EfficientNetV2B0 model
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras.models import Sequential
from tensorflow.keras import models
# Create a sequential model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
# Extract features from image
features = {}
directory = BASE_DIR
for img_name in tqdm(os.listdir(directory)):
    # Load the image from file
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features
    feature = model.predict(image, verbose=0)

    # Get image ID
    image_id = img_name.split('.')[0]

    # Store feature
    features[image_id] = feature

# Store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
# Load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('best_model.h5')
# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Function to generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace("startseq ", "").replace(" endseq", "")

from PIL import Image
import matplotlib.pyplot as plt

def generate_caption(image_name):
    # Load the image
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, image_name)
    image = Image.open(img_path)

    # Predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print('Predicted Caption: ', y_pred, "\n")


# Streamlit App
st.title("Deep Learning Image Captioning App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_path = os.path.join(directory, uploaded_file.name)

    # Save the uploaded image
    image.save(img_path)

    # Button to generate caption
    if st.button("Generate Caption"):
        # Preprocess the image
        img_array = img_to_array(image)
        img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

        # Generate and display caption
        caption = predict_caption(model, features[uploaded_file.name.split('.')[0]], tokenizer, max_length)
        st.write("Predicted Caption:", caption)
