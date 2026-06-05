# filters.py
import cv2
import numpy as np
def filter_contours_by_area(contours, img_shape,
                    min_area=100, max_area_ratio=0.3,
                    min_aspect=0.2, max_aspect=5.0):
    h, w = img_shape[:2]
    max_area = h * w * max_area_ratio
    mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / (bw + 1e-5)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        # 近似多边形
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # 太简单的可能是箭头
        if len(approx) < 4:
            continue
        cv2.fillPoly(mask, [cnt], 255)
    return mask


