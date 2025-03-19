import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load the model and image processor
processor = AutoImageProcessor.from_pretrained("beingamit99/car_damage_detection")
model = AutoModelForImageClassification.from_pretrained("beingamit99/car_damage_detection")

# Define the function that takes an image as input and returns a text output
def classify_image(input_image):
    # Load and process the image
    image = np.array(input_image)
    inputs = processor(images=image, return_tensors="pt")

    # Make predictions
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    predicted_class_id = np.argmax(logits)
    predicted_proba = np.max(logits)
    label_map = model.config.id2label
    predicted_class_name = label_map[predicted_class_id]

    # OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+3, y1:y2+3]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text = result[0][1]

    # Return the results
    return f"Predicted class: {predicted_class_name} (probability: {predicted_proba:.4f}", text

# Create Gradio interface
input_image = gr.components.Image()
output_text = gr.components.Text()
output_text2 = gr.components.Text()

gr.Interface(fn=classify_image, inputs=input_image, outputs=[output_text, output_text2], title="AutoVision").launch(debug = 1)