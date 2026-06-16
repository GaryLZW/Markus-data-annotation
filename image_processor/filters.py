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
        return cnt[::1]
    
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center = np.array([cx, cy])
    pts = cnt.astype(np.float32)
    
    pts_uniform = cnt[::uniform_step] 


    dists = np.linalg.norm(pts - center, axis=1)
    # 距离权重
    probs = dists / dists.sum()
    
    replace = False
    if n_samples < len(cnt) * 0.1:
        n_samples = int(len(cnt) * 0.1)
    elif n_samples > len(cnt)-5:
        n_samples = len(cnt) - 1
        replace = True
    
    # print(len(cnt))
    idx_weighted = np.random.choice(
        len(cnt),
        size=n_samples,
        replace=replace,
        p=probs
    )
    
    sampled = np.vstack([pts_uniform, pts[idx_weighted]])
    
    # 强制加入最远的两个端点
    if boost_endpoints:
        far_idx = np.argsort(dists)[-2:]
        sampled = np.vstack([sampled, pts[far_idx]])
    
    return sampled



def sample_skeleton_points(img, n_samples=30):
    # Convert original image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    skel = cv2.ximgproc.thinning(thresh)

    pts = np.column_stack(np.where(skel > 0))
    
    if len(pts) == 0:
        return np.empty((0, 2))
    
    if n_samples < len(pts) * 0.2:
        n_samples = int(len(pts) * 0.2)
    elif n_samples > len(pts)-5:
        n_samples = len(pts) - 1
    # # 重心
    # center = pts.mean(axis=0)
    # # 距离权重
    # dists = np.linalg.norm(pts - center, axis=1)
    # probs = dists / dists.sum()
    # idx = np.random.choice(
    #                        len(pts),
    #                        size=min(n_samples, len(pts)),
    #                        replace=False,
    #                        p=probs
    # )
    
    # skel_pts = np.array([(x, y) for (y, x) in pts[idx]])
    skel_pts = np.array([(x, y) for (y, x) in pts[::int(len(pts)/n_samples)]])
    return skel_pts

def assign_skeleton_to_contours(skel_pts, contours, max_dist=10):
    """
    返回：
    skeleton_points: Nx2
    skeleton_owner: N (轮廓索引)
    """
    skel_pts = skel_pts.astype(np.uint8)
    
    return np.repeat(-1, len(skel_pts))
    # owner = []

    # for pt in skel_pts:

    #     best_i = -1
    #     best_d = max_dist

    #     for i, cnt in enumerate(contours):
    #         d = cv2.pointPolygonTest(cnt, pt, True)
    #         if d >= 0 and d < best_d:
    #             best_d = d
    #             best_i = i

    #     owner.append(best_i)

    # return np.array(owner)

def extract_point_cloud(img, contours, uniform_step, far_points_per_contour, use_endpoint_boost):
    """
    返回：
    - points: Nx2
    - point_map: 每个点对应哪个轮廓
    """
    # Two sets of points Contour Skeleton
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
    
    s_pts = sample_skeleton_points(img)
    
    skel_owner = assign_skeleton_to_contours(s_pts, contours)
    for j, p in enumerate(s_pts):
        points.append(p.tolist())
        point_map.append( skel_owner[j] )
    
    return np.array(points), point_map

