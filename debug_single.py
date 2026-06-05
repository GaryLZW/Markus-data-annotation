import cv2
import numpy as np
from image_processor.detectors import extract_structure_mask
from image_processor.filters import filter_contours_by_area

#IMG_PATH = "dataset/raw/mol_0216.png"  

def debug_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(":x: 图片读取失败")
        return

    # 1. 边缘掩膜
    edge_mask = extract_structure_mask(
        img,
        blur_ksize=5,
        canny_low=50,
        canny_high=150
    )

    # 2. 轮廓
    contours, _ = cv2.findContours(
        edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    

    structure_mask = filter_contours_by_area(
        contours,
        img.shape,
        min_area=300,
        max_area_ratio=0.3,
        min_aspect=0.2,
        max_aspect=5.0
    )

    # 3. 应用掩膜
    masked = cv2.bitwise_and(img, img, mask=structure_mask)

    # 4. 二值化
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ===== 显示调试窗口 =====
    cv2.imshow("1. Original", img)
    cv2.imshow("2. Edge Mask", edge_mask)
    cv2.imshow("3. Structure Mask", structure_mask)
    cv2.imshow("4. Final Binary", binary)

    print(":point_right: 按任意键继续下一张，Q 退出")

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return key == ord('q')

if __name__ == "__main__":
    test_images = [
        "dataset/raw/mol_0216.png",
        "dataset/raw/mol_0220.jpg",
        "dataset/raw/mol_0222.png",
        # 多加几张典型样本
    ]

    for path in test_images:
        IMG_PATH = path
        if debug_pipeline(IMG_PATH):
            break
