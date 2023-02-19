import os
import cv2
from model import model

IMAGE_PATH = r"C:\Users\hadar\Desktop\Projects\MAFAT-Challenge\testing\data\images\126_1280_5120.tiff"

# Load the model
model = model()
model.load(r"C:\Users\hadar\Desktop\Projects\MAFAT-Challenge")

# Load the image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Predict the objects in the image
detections = model.predict(img)

# Parse the output of the predict function to extract the detection information
for i, cls in enumerate(model.CLASSES):
    for detection in detections[i]:
        confidence = detection[0]
        coordinates = detection[1:]
        x1, y1, x2, y2, x3, y3, x4, y4 = coordinates
        # Do something with the detection information
