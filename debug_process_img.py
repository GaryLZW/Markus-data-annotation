import cv2
from image_processor.process import process_one_image
import os



if __name__ == "__main__":
    test_images = [
        "dataset/raw/mol_0216.png",
        "dataset/raw/mol_0220.jpg",
        "dataset/raw/mol_0222.png",
        "dataset/raw/mol_0141.png",
    ]
    test_images = [
        "dataset/raw/"+f for f in os.listdir("dataset/raw")
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    for i, path in enumerate(test_images):
        process_one_image(path, f"dataset/processed/test{i}.png", invert=True)
