import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Define the path to the trained model
model_path = 'app/trained_model/trained_oxford_flowers_model.h5'  # Adjust this path as per your file location in Colab

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class labels for Oxford Flowers 102 dataset
class_names = [
    'African Violet', 'Alpine aster', 'American bluebell', 'Anemone', 
    'Apple', 'Asian lily', 'Astilbe', 'Aubrieta', 'Bee balm', 
    'Bittercress', 'Black-eyed Susan', 'Bluebell', 'Buttercup', 
    'Cabbage', 'Cactus', 'Canola', 'Carnation', 'Chrysanthemum', 
    'Clematis', 'Columbine', 'Common daisy', 'Common foxglove', 
    'Common marigold', 'Common snapdragon', 'Common yarrow', 
    'Cosmos', 'Crown daisy', 'Daffodil', 'Dahlia', 
    'Delphinium', 'Dianthus', 'Echinacea', 'Flowering maple', 
    'Foxglove', 'Garden lily', 'Geranium', 'Giant hyssop', 
    'Ginger', 'Gladiolus', 'Goldenrod', 'Hollyhock', 
    'Horehound', 'Iberis', 'Iris', 'Lantana', 
    'Larkspur', 'Lavender', 'Lilac', 'Lily', 
    'Lobelia', 'Lotus', 'Marigold', 'Michaelmas daisy', 
    'Morning glory', 'Nasturtium', 'Orchid', 'Pansy', 
    'Peony', 'Petunia', 'Poppy', 'Primula', 
    'Ragged robin', 'Rhododendron', 'Rose', 'Sage', 
    'Scabious', 'Snapdragon', 'Sunflower', 'Sweet pea', 
    'Tulip', 'Verbena', 'Violet', 'Wallflower', 
    'Wild geranium', 'Wild sweet pea', 'Wisteria', 
    'Zinnia'
    # Add the remaining class labels to make a total of 102
]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((150, 150))  # Resize to the input size expected by your model
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape((1, 150, 150, 3))  # Reshape to (1, height, width, channels)
    return img_array

# Streamlit App
st.title('Oxford Flowers Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')
