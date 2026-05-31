import cv2
import numpy as np
from detectors import extract_structure_mask
from filters import filter_contours
from enhancer import enhance_image  # 后面给
def process_one_image(img_path, out_path):
    img = cv2.imread(img_path)
    # 1. 提取边缘掩膜
    edge_mask = extract_structure_mask(img)
    # 2. 找轮廓并筛选
    contours, _ = cv2.findContours(
        edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    structure_mask = filter_contours(contours, img.shape)
    # 3. 应用掩膜
    result = cv2.bitwise_and(img, img, mask=structure_mask)
    # 4. 转灰度 + 二值化（给 OCR 用）
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # 5. 超分辨率增强（可选）
    enhanced = enhance_image(binary)
    cv2.imwrite(out_path, enhanced)

