import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from image_processor.detectors import extract_structure_mask
from image_processor.filters import filter_contours_by_area, distance_weighted_sample, extract_point_cloud  
from collections import Counter
from image_processor.img_preprocess import upscale, preprocess, is_this_jpg

# ===== 可调参数=====

EPS = 40                 # 聚类半径
MIN_SAMPLES = 70        # 最小点数
uniform_step = 3
far_points_per_contour = 10   # 远端采样数
use_endpoint_boost = True    # 是否强制加入最远端点

# -----------------------------
# 工具函数
# -----------------------------

def draw_point_cloud(img, points, labels):
    vis = img.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    for i, p in enumerate(points):
        label = labels[i]
        if label == -1:
            color = (100, 100, 100)
        else:
            color = colors[label % len(colors)]
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, color, -1)
    return vis

# -----------------------------
# 主调试流程
# -----------------------------

def debug_distance_weighted_cluster(img_path):
    is_jpg = is_this_jpg(img_path)
    ksize = 14

    img = cv2.imread(img_path)
    if img is None:
        print(":x: 图片读取失败")
        return

    # 1. Check img quality
    
    
    h, w = img.shape[:2]

    if h < 1500:
        img = upscale( img, scale=int(1500/h) )

    edges = extract_structure_mask(img, vis=True)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    m00 = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        M = cv2.moments(cnt)
        m00.append(M["m00"] == 0) # Is this contour very small


    if np.any(m00):
        edges = preprocess(img, ksize, is_jpg)
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
    points, point_map = extract_point_cloud(img, contours, uniform_step, far_points_per_contour, use_endpoint_boost)
    

    # 3. DBSCAN 聚类
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(points)
    
    
    # 3. HDBSCAN 聚类
    #db = HDBSCAN(min_cluster_size=MIN_SAMPLES, min_samples=MIN_SAMPLES)
    #labels = db.fit_predict(points)


    # 4. 可视化点云
    cloud_vis = draw_point_cloud(img, points, labels)
    cv2.imshow("1. Distance-Weighted Point Cloud", cloud_vis)
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
            contour_labels[point_map[i]] = main_label # NOT -1
    
    # 7. 主聚类轮廓 -- 主聚类轮廓中的次聚类点 -- 相关次聚类轮廓
    main_contour_idx = [i for i, cnt in enumerate(contours) if contour_labels[i] == main_label]
    
    sub_cluster_lablels = [lbl for i, lbl in enumerate(labels) if (point_map[i] in main_contour_idx)]
    
    for i, lbl in enumerate(labels):
        if lbl in sub_cluster_lablels:
            contour_labels[point_map[i]] = lbl # NOT -1


    # 8. 生成掩膜
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, cnt in enumerate(contours):
        if contour_labels[i] != -1:
            cv2.fillPoly(mask, [cnt], 255)
    
    masked = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    cv2.imshow("2. Main Structure Mask", mask)
    cv2.imshow("3. Final Result", gray)
    print(":point_right: 按任意键继续，Q 退出")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key == ord('q')
    
    

if __name__ == "__main__":
    test_images = [
        "dataset/raw/"+f for f in os.listdir("dataset/raw")
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    test_images = [
        "dataset/raw/mol_0101.png",
    ]
    for path in test_images:
        #if debug_clustering(path):
        if debug_distance_weighted_cluster(path):
            break
