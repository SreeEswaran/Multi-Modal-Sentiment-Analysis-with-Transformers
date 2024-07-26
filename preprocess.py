import pandas as pd
from PIL import Image
import os

def preprocess_text_data(text_file):
    df = pd.read_csv(text_file)  
    return df

def preprocess_image_data(image_folder):
    images = []
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path)
        images.append(image)
    return images
