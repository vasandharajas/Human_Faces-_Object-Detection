import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os
import cv2

# Define the custom metric
def custom_metric(y_true, y_pred):
    return tf.keras.metrics.MeanSquaredError()(y_true, y_pred)

# Load dataset
@st.cache_data
def load_data():
    image_path = "C:/Users/anand/Desktop/Final Project1 Human Face Hugging/images/images"
    faces_data = pd.read_csv("C:/Users/anand/Desktop/Final Project1 Human Face Hugging/faces.csv")

    data = faces_data.copy()
    data["image_name"] = data["image_name"].apply(lambda x: os.path.join(image_path, x))
    
    # Normalizing x0, y0, x1, y1
    data["x0"] = data["x0"] / data["width"]
    data["y0"] = data["y0"] / data["height"]
    data["x1"] = data["x1"] / data["width"]
    data["y1"] = data["y1"] / data["height"]
    return data

# Load the model with custom objects
@st.cache_resource
def load_model_with_custom_objects():
    with tf.keras.utils.custom_object_scope({'custom_metric': custom_metric}):
        return tf.keras.models.load_model("my_model.keras")

# Preprocessing function
def preprocessing(file_path, label=None, pred=False):
    img = cv2.imread(file_path)
    height_img, width_img, _ = img.shape
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    if pred:
        return img, width_img, height_img
    return img, label

# Predict and visualize multiple images
def predict_and_visualize_multiple(img_paths, model):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(img_paths):
        # Preprocess image
        test_img, width, height = preprocessing(img_path, None, pred=True)
        test_img = np.expand_dims(test_img, axis=0)
        
        # Make prediction
        pred = model.predict(test_img)
        
        # Open the image using OpenCV
        imr = cv2.imread(img_path)
        imr = cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
        
        # Back to original image dimensions
        x0 = int(pred[0][0] * width)
        y0 = int(pred[0][1] * height)
        x1 = int(pred[0][2] * width)
        y1 = int(pred[0][3] * height)
        
        # Draw rectangle on image
        color = (255, 0, 0)  # Red color
        thickness = 3
        imr = cv2.rectangle(imr, (x0, y0), (x1, y1), color, thickness)
        
        # Display the image in a grid
        axes[i].imshow(imr)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i + 1}')
    
    plt.tight_layout()
    st.pyplot(fig)

# Predict and visualize a single image
def predict_and_visualize_single(img_path, model):
    # Preprocess image
    img, width, height = preprocessing(img_path, pred=True)
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction
    pred = model.predict(img)
    
    # Open the image using OpenCV
    imr = cv2.imread(img_path)
    imr = cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
    
    # Back to original image dimensions
    x0 = int(pred[0][0] * width)
    y0 = int(pred[0][1] * height)
    x1 = int(pred[0][2] * width)
    y1 = int(pred[0][3] * height)
    
    # Draw a rectangle on the image
    color = (255, 0, 0)  # Red color
    thickness = 3
    imr = cv2.rectangle(imr, (x0, y0), (x1, y1), color, thickness)
    
    # Display the image
    st.image(imr, caption="Predicted Bounding Box", use_column_width=True)

# Load the model
model = load_model_with_custom_objects()

# Streamlit UI
def main():
    st.title("Model Prediction And Image Dataset")

    st.sidebar.title("In This Project")
    option = st.sidebar.selectbox("Choose to display", ["Dataset", "Model Metrics", "Predictions"])

    data = load_data()

    if option == "Dataset":
        st.header("Dataset Used for Model Building")
        st.write(data.head())
    
    elif option == "Model Metrics":
        st.header("Model Performance Metrics")
        # Assume `history` is a global variable containing training history
        # For simplicity, using random values here
        history = {
            'accuracy': [0.8, 0.85],
            'val_accuracy': [0.75, 0.80],
            'loss': [0.5, 0.4],
            'val_loss': [0.6, 0.5]
        }
        st.subheader("Training & Validation Accuracy")
        fig, ax = plt.subplots()
        ax.plot(history['accuracy'], label='Train Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Training & Validation Loss")
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss')
        ax.legend()
        st.pyplot(fig)
    
    elif option == "Predictions":
        st.header("Predictions on Sample Images")
        image_folder = 'C:/Users/anand/Desktop/Final Project1 Human Face Hugging/images/images'
        image_files = os.listdir(image_folder)
        image_paths = [os.path.join(image_folder, img_file) for img_file in image_files[:10]]  
        predict_and_visualize_multiple(image_paths, model)

    st.title('Model Prediction')
    
    # Example input
    img_path = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if img_path:
        # Create the temporary directory if it doesn't exist
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save the uploaded file temporarily
        temp_img_path = os.path.join(temp_dir, img_path.name)
        with open(temp_img_path, "wb") as f:
            f.write(img_path.getbuffer())
        
        # Perform prediction and visualization
        predict_and_visualize_single(temp_img_path, model)

if __name__ == "__main__":
    main()

