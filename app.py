import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained YOLO model
model = YOLO('best.pt')

# Streamlit app title
st.title('YOLOv8 Object Detection and Measurement App')

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def detect_and_measure(image):
    # Convert image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Run YOLO model on the uploaded image
    results = model(image_cv)
    
    # Get results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    class_labels = results[0].names  # Class labels
    
    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        # Convert dimensions from pixels to inches
        dpi = 96
        width_in_inches = width / dpi
        height_in_inches = height / dpi
        
        # Get the label of the detected object
        object_label = class_labels[class_id]
        
        # Print dimensions and label
        st.write(f"Object: {object_label}, Width of Box: {width_in_inches:.2f} inches, Height of Box: {height_in_inches:.2f} inches")
        
        # Annotate image with dimensions and label
        plt.text(x1, y1 - 10, f'{object_label}: {width_in_inches:.2f} inches x {height_in_inches:.2f} inches',
                 color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    
    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Run detection and measurement
    result_buf = detect_and_measure(image)
    
    # Display the results using Streamlit
    st.image(result_buf, caption='Detection Results.', use_column_width=True)
