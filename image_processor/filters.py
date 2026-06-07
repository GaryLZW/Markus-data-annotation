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

def distance_weighted_sample(cnt, uniform_step=3, n_samples=8, boost_endpoints=True):
    """
    按到中心点的距离加权采样轮廓点
    加轮廓均匀采点
    """
    cnt = cnt.reshape(-1, 2)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return cnt[::len(cnt)//max(n_samples,1)]
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center = np.array([cx, cy])
    pts = cnt.astype(np.float32)
    
    pts_uniform = cnt[::uniform_step] 


    dists = np.linalg.norm(pts - center, axis=1)
    # 距离权重
    probs = dists / dists.sum()
    
    if n_samples < len(cnt) * 0.3:
        n_samples = int(len(cnt) * 0.3)
    elif n_samples > len(cnt)-5:
        n_samples = len(cnt) - 1
    

    idx_weighted = np.random.choice(
        len(cnt),
        size=n_samples,
        replace=False,
        p=probs
    )
    
    sampled = np.vstack([pts_uniform, pts[idx_weighted]])
    
    # 强制加入最远的两个端点
    if boost_endpoints:
        far_idx = np.argsort(dists)[-2:]
        sampled = np.vstack([sampled, pts[far_idx]])
    
    return sampled


def extract_point_cloud(contours, uniform_step, far_points_per_contour, use_endpoint_boost):
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
            uniform_step,
            far_points_per_contour,
            use_endpoint_boost
        )
        for p in far_pts:
            points.append(p.tolist())
            point_map.append(i)
    return np.array(points), point_map

