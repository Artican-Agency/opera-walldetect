import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load and resize the image
image_path = r'C:\Users\lenovo\Desktop\Wall_Detection\images\blog-30.jpg'
image = cv2.imread(image_path)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Resize for easier visualization
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
resized_hsv = cv2.resize(image_hsv, dim, interpolation=cv2.INTER_AREA)

# Define HSV color range for the wall (adjust if needed)
lower_orange = np.array([5, 70, 70])
upper_orange = np.array([25, 255, 255])

# Create a mask for the wall based on its color
initial_mask = cv2.inRange(resized_hsv, lower_orange, upper_orange)

# Prepare initial mask for GrabCut
grabcut_mask = np.zeros(resized_image.shape[:2], np.uint8)
grabcut_mask[initial_mask > 0] = cv2.GC_PR_FGD  # Probable foreground (wall)
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

# Apply GrabCut to refine wall segmentation
cv2.grabCut(resized_image, grabcut_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
refined_wall_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
wall_mask = refined_wall_mask * 255

# Define the new wall color (choose any RGB color you like)
# Example palette colors
palette = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "purple": (128, 0, 128),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255)
}

# Choose a color for the wall (you can change this value)
selected_color_name = "purple"
selected_color = palette[selected_color_name]

# Create a colored overlay for the wall
colored_wall = np.zeros_like(resized_image, np.uint8)
colored_wall[:] = selected_color

# Apply the mask to color only the wall
colored_wall_masked = cv2.bitwise_and(colored_wall, colored_wall, mask=wall_mask)

# Combine the colored wall with the original image
result = cv2.addWeighted(resized_image, 1, colored_wall_masked, 0.6, 0)

# Display results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Wall Mask
plt.subplot(1, 3, 2)
plt.imshow(wall_mask, cmap='gray')
plt.title("Wall Mask")
plt.axis("off")

# Result with Colored Wall
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f"Wall Colored with {selected_color_name}")
plt.axis("off")

plt.tight_layout()
plt.show()
