import os

from helpers import frame_with_annotation

IMAGES_PATH = r"C:\Users\hadar\Desktop\Projects\MAFAT-Challenge\testing\data\images"

try:
  for root, dirs, files in os.walk(IMAGES_PATH):
    for file in files:
      image_path = os.path.join(root, file)
      frame_with_annotation(image_path)
except Exception as err:
  print(f"Images display has failed, error: {err}")
