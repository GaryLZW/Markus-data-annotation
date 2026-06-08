import cv2
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from image_processor.detectors import extract_structure_mask
from image_processor.filters import filter_contours_by_area, distance_weighted_sample, extract_point_cloud  
from collections import Counter

def distance_weighted_cluster(img_path, eps, min_samples, uniform_step, far_points_per_contour, use_endpoint_boost=True):
    img = cv2.imread(img_path)
    if img is None:
        print(":x: 图片读取失败")
        return
    # 1. 边缘 + 初筛
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #edges = cv2.Canny(blurred, 50, 150)
    
    edges = extract_structure_mask(img)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    '''
    contours = [
        cnt for cnt in contours
        if 80 < cv2.contourArea(cnt) < img.shape[0]*img.shape[1]*0.35
    ]
    '''
    if len(contours) == 0:
        print(":warning: 未检测到轮廓")
        return
    # 2. 构建点云
    points, point_map = extract_point_cloud(contours, uniform_step, far_points_per_contour, use_endpoint_boost)
    

    # 3. DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    
    
    # 3. HDBSCAN 聚类
    #db = HDBSCAN(min_cluster_size=MIN_SAMPLES, min_samples=MIN_SAMPLES)
    #labels = db.fit_predict(points)


    # 4. 可视化点云
    #cloud_vis = draw_point_cloud(img, points, labels)
    #cv2.imshow("1. Distance-Weighted Point Cloud", cloud_vis)
    # 5. 找主聚类
    label_counts = Counter(labels)
    label_counts.pop(-1, None)
    if not label_counts:
        print(":warning: 未找到主结构聚类")
        return
    main_label = max(label_counts, key=label_counts.get)
    # 6. 主聚类点 → 主聚类轮廓
    contour_labels = np.full(len(contours), -1)
    for i, lbl in enumerate(labels):
        if lbl == main_label:
            contour_labels[point_map[i]] = main_label
    
    # 7. 主聚类轮廓 -- 主聚类轮廓中的次聚类点 -- 相关次聚类轮廓
    main_contour_idx = [i for i, cnt in enumerate(contours) if contour_labels[i] == main_label]
    
    sub_cluster_lablels = [lbl for i, lbl in enumerate(labels) if point_map[i] in main_contour_idx]
    
    for i, lbl in enumerate(labels):
        if lbl in sub_cluster_lablels:
            contour_labels[point_map[i]] = lbl


    # 8. 生成掩膜
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, cnt in enumerate(contours):
        if contour_labels[i] != -1:
            cv2.fillPoly(mask, [cnt], 255)
    
    masked = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("2. Main Structure Mask", mask)
    #cv2.imshow("3. Final Result", masked)

    return masked
    

def save_for_ocr(mask, out_path, invert=True, pad=10):
    """
    mask: 结构式掩膜（0/255）
    """
    # 可选反色
    if invert:
        roi = cv2.bitwise_not(mask)
    else:
        roi = mask
    # 加 padding
    roi = cv2.copyMakeBorder(
        roi, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=255 if invert else 0
    )
    cv2.imwrite(out_path, roi)


def process_one_image(img_path, out_path, invert=False):
    # ===== 可调参数=====

    EPS = 100                 # 聚类半径
    MIN_SAMPLES = 7        # 最小点数
    uniform_step = 3
    far_points_per_contour = 10   # 远端采样数
    use_endpoint_boost = True    # 是否强制加入最远端点

    masked = distance_weighted_cluster(img_path, EPS, MIN_SAMPLES, uniform_step, far_points_per_contour, use_endpoint_boost)

    save_for_ocr(masked, out_path, invert)


