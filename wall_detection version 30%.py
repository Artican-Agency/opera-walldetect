import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser

# Global variables
image = None
contours = []
original_image = None

# Function to open a color picker and apply the chosen color to the clicked wall
def select_color_and_recolor(x, y):
    global image, contours

    # Check which contour was clicked
    for i, contour in enumerate(contours):
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            # Open the color picker
            color = colorchooser.askcolor()[0]
            if color:
                # Convert the color from RGB to BGR (for OpenCV)
                color_bgr = (int(color[2]), int(color[1]), int(color[0]))

                # Create a mask for the clicked wall
                mask = np.zeros_like(gray_image, dtype=np.uint8)  # Ensure it's 8-bit
                cv2.drawContours(mask, contours, i, 255, thickness=cv2.FILLED)  # Fill contour in mask

                # Recolor the wall
                recolor_wall(mask, color_bgr)
            break

# Function to recolor the wall using blending for a natural effect
def recolor_wall(mask, color_bgr):
    global image, original_image

    # Convert the mask to 3 channels (for BGR image)
    mask_3channel = cv2.merge([mask, mask, mask])

    # Create an image filled with the chosen color
    colored_wall = np.zeros_like(image)
    colored_wall[:] = color_bgr

    # Blend the original image with the recolored wall
    recolored_image = cv2.addWeighted(original_image, 0.6, colored_wall, 0.4, 0)

    # Apply the mask to recolor only the wall
    image = np.where(mask_3channel == 255, recolored_image, image)

    # Display the updated image
    cv2.imshow('Recolor Walls', image)

# Mouse callback function to handle clicks
def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        select_color_and_recolor(x, y)

# Load and resize the image
image = cv2.imread(r"C:\Users\lenovo\Desktop\Wall_Detection\images\bases-de-la-decoration-interieur.jpg")
original_image = cv2.resize(image.copy(), (600, 400))
image = cv2.resize(image, (600, 400))

# Convert to grayscale and apply adaptive thresholding for better wall detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Find contours (walls)
contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the image with detected walls
cv2.imshow('Recolor Walls', contour_image)

# Set the mouse callback to capture clicks
cv2.setMouseCallback('Recolor Walls', mouse_click_event)

# Wait until the user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

