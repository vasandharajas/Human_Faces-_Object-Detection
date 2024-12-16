import os
import torch
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import cv2

# Load dataset
@st.cache_data
def load_data():
    image_path = "C:/Users/anand/Desktop/Final Project1 Human Face Hugging/images/images"
    csv_path = "C:/Users/anand\Desktop/Final Project1 Human Face Hugging/faces.csv"
    # Print the paths for debugging
    print(f"Loading images from: {image_path}")
    print(f"Loading CSV from: {csv_path}")
    
    try:
        faces_data = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        st.error(f"File not found: {csv_path}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    data = faces_data.copy()
    data["image_name"] = data["image_name"].apply(lambda x: os.path.join(image_path, x))
    
    # Normalizing x0, y0, x1, y1
    data["x0"] = data["x0"] / data["width"]
    data["y0"] = data["y0"] / data["height"]
    data["x1"] = data["x1"] / data["width"]
    data["y1"] = data["y1"] / data["height"]
    return data

# Load YOLOv5 model
@st.cache_resource
def load_yolov5_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Preprocessing function
def preprocessing(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Predict and visualize multiple images
def predict_and_visualize_multiple(img_paths, model):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(img_paths):
        img = preprocessing(img_path)
        
        # Make prediction
        results = model(img)
        results.render()  # updates results.imgs with boxes and labels
        
        # Display the image in a grid
        axes[i].imshow(results.imgs[0])
        axes[i].axis('off')
        axes[i].set_title(f'Image {i + 1}')
    
    plt.tight_layout()
    st.pyplot(fig)

# Predict and visualize a single image
def predict_and_visualize_single(img_path, model):
    img = preprocessing(img_path)
    
    # Make a prediction
    results = model(img)
    results.render()  # updates results.imgs with boxes and labels
    
    # Display the image
    st.image(results.imgs[0], caption="Predicted Bounding Box", use_column_width=True)

# Load the model
model = load_yolov5_model()

# Streamlit UI
def main():
    st.title("YOLOv5 Model Prediction And Image Dataset")

    st.sidebar.title("In This Project")
    option = st.sidebar.selectbox("Choose to display", ["Dataset", "Model Metrics", "Predictions"])

    data = load_data()

    if data.empty:
        st.warning("Data could not be loaded. Please check the file path.")
        return

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
