# detectors.py
import cv2
import numpy as np

def extract_structure_mask(img, blur_ksize=5, canny_low=50, canny_high=150):
    """
    返回结构式的二值掩膜
    """
    # Convert original image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去噪
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # 膨胀边缘，连起虚线
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges
