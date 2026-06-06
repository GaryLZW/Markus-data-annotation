import cv2
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from image_processor.detectors import extract_structure_mask
from image_processor.filters import filter_contours_by_area  
from collections import Counter

# ===== 可调参数=====

EPS = 100                 # 聚类半径
MIN_SAMPLES = 7        # 最小点数
far_points_per_contour = 10   # 远端采样数
use_endpoint_boost = True    # 是否强制加入最远端点

# -----------------------------
# 工具函数
# -----------------------------

def draw_clusters(img, contours, labels):
    """
    给不同聚类上色
    """
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    vis = img.copy()
    for i, cnt in enumerate(contours):
        color = (100, 100, 100)  # 默认灰色（噪声）
        if labels[i] != -1:
            color = colors[labels[i] % len(colors)]
        cv2.drawContours(vis, [cnt], -1, color, 2)
        # 画中心点
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 3, color, -1)
    return vis

def distance_weighted_sample(cnt, n_samples=8, boost_endpoints=True):
    """
    按到中心点的距离加权采样轮廓点
    """
    cnt = cnt.reshape(-1, 2)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return cnt[::len(cnt)//max(n_samples,1)]
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center = np.array([cx, cy])
    pts = cnt.astype(np.float32)
    dists = np.linalg.norm(pts - center, axis=1)
    # 距离权重
    probs = dists / dists.sum()
    
    if n_samples < len(cnt) * 0.4:
        n_samples = int(len(cnt) * 0.4)
    elif n_samples > len(cnt):
        n_samples = len(cnt) - 1
    

    idx = np.random.choice(
        len(cnt),
        size=n_samples,
        replace=False,
        p=probs
    )
    sampled = pts[idx]
    # 强制加入最远的两个端点
    if boost_endpoints:
        far_idx = np.argsort(dists)[-2:]
        sampled = np.vstack([sampled, pts[far_idx]])
    return sampled


def extract_point_cloud(contours, far_points_per_contour, use_endpoint_boost):
    """
    返回：
    - points: Nx2
    - point_map: 每个点对应哪个轮廓
    """
    points = []
    point_map = []
    for i, cnt in enumerate(contours):
        # 中心点
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            points.append([cx, cy])
            point_map.append(i)
        # 距离加权远端采样
        far_pts = distance_weighted_sample(
            cnt,
            far_points_per_contour,
            use_endpoint_boost
        )
        for p in far_pts:
            points.append(p.tolist())
            point_map.append(i)
    return np.array(points), point_map


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
    img = cv2.imread(img_path)
    if img is None:
        print(":x: 图片读取失败")
        return
    # 1. 边缘 + 初筛
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
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
    points, point_map = extract_point_cloud(contours, far_points_per_contour, use_endpoint_boost)
    

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
    cv2.imshow("2. Main Structure Mask", mask)
    cv2.imshow("3. Final Result", masked)
    print(":point_right: 按任意键继续，Q 退出")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key == ord('q')

def debug_clustering(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(":x: 图片读取失败")
        return
    # 1. 边缘 + 初步面积筛选
    edge_mask = extract_structure_mask(img,
                                       blur_ksize=5,
                                       canny_low=50,
                                       canny_high=150
                                      )
    contours, _ = cv2.findContours(
        edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    '''
    # 先用面积粗筛（减少噪声）
    structure_mask = filter_contours_by_area(
        contours,
        img.shape,
        min_area=300,
        max_area_ratio=0.3,
        min_aspect=0.2,
        max_aspect=5.0
    )
    
    #  应用掩膜
    masked = cv2.bitwise_and(img, img, mask=structure_mask)
    
    # 二值化
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    contours, _ = cv2.findContours(
        structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    '''
    # 2. 计算轮廓中心
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append([cx, cy])
    if len(centroids) == 0:
        print(":warning: 没有检测到轮廓")
        return
    else:
        print("CONTOUR CENTROID DETECTED")

    # 3. DBSCAN 聚类
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(np.array(centroids))
    
    # 4. 可视化聚类
    cluster_vis = draw_clusters(img, contours, labels)
    cv2.imshow("1. Clustering Result", cluster_vis)
    
    
    # 5. 找主聚类
    
    from collections import Counter
    
    label_counts = Counter(labels)
    label_counts.pop(-1, None)
    if not label_counts:
        print(":warning: 未找到主结构聚类")
        return
    main_label = max(label_counts, key=label_counts.get)
    
    
    # 6. 生成最终掩膜
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, cnt in enumerate(contours):
        if labels[i] == main_label:
            cv2.fillPoly(mask, [cnt], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("2. Main Structure Mask", mask)
    cv2.imshow("3. Final Result", masked)
    print(":point_right: 按任意键继续，Q 退出")
    
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return key == ord('q')

if __name__ == "__main__":
    test_images = [
        "dataset/raw/mol_0216.png",
        "dataset/raw/mol_0220.jpg",
        "dataset/raw/mol_0222.png",
    ]
    for path in test_images:
        #if debug_clustering(path):
        if debug_distance_weighted_cluster(path):
            break
