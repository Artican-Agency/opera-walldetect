import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import colorchooser

# Function to load the DeepLabV3 model
def load_model():
    # Use a pre-trained DeepLabV3 model with weights for segmentation tasks
    return tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)

# Function to preprocess the image for the model
def preprocess_image(image):
    input_image = cv2.resize(image, (513, 513))  # Resize to 513x513
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = input_image / 255.0  # Normalize to [0, 1]
    return input_image

# Function to perform semantic segmentation using the DeepLabV3 model
def segment_image(model, image):
    input_tensor = preprocess_image(image)
    # Predict the segmentation mask
    predictions = model.predict(input_tensor)
    predictions = tf.argmax(predictions[0], axis=-1).numpy()

    # Resize the prediction back to the original image size
    mask = cv2.resize(predictions, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

# Function to color the walls based on the segmentation mask
def recolor_walls(image, mask):
    global recolored_image

    # Choose a color using a color picker
    color = colorchooser.askcolor()[0]
    if color:
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # Convert RGB to BGR

        # Apply the color only to the wall region (label 15 is for 'wall' in DeepLab)
        recolored_image = image.copy()
        recolored_image[mask == 15] = color_bgr  # Assuming '15' corresponds to walls

        # Display the recolored image
        cv2.imshow('Recolored Walls', recolored_image)

# Mouse event callback function for recoloring walls
def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        recolor_walls(image, segmentation_mask)

# Load your image
image_path = r'C:\Users\lenovo\Desktop\Wall_Detection\images\blog-30.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (600, 400))

# Load the DeepLabV3 model
model = load_model()

# Perform semantic segmentation
segmentation_mask = segment_image(model, image)

# Show the segmented mask (optional)
cv2.imshow('Segmentation Mask', segmentation_mask * 15)  # Multiplied to enhance visualization

# Set up mouse click event for recoloring walls
cv2.setMouseCallback('Recolored Walls', mouse_click_event)

# Display the original image
cv2.imshow('Recolored Walls', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
