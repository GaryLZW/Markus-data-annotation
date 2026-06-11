#preprocessing for jpg s
import cv2

def is_this_jpg(img_path):
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return True
    else:
        return False

def upscale(img, scale=2):
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

def remove_jpeg_blocks(img, strength=3):
    """
    strength: 3~5，太大会糊
    """
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=strength,
        hColor=strength,
        templateWindowSize=7,
        searchWindowSize=21
    )

def guided_enhance(img, radius=5, eps=1e-2, is_jpg=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_jpg:
        guided = cv2.ximgproc.guidedFilter(
            guide=gray,
            src=gray,
            radius=radius,
            eps=eps
        )
    else:
        guided = cv2.ximgproc.guidedFilter(
            guide=gray,
            src=gray,
            radius=3,
            eps=1e-2
        )
    return guided

def connect_broken_edges(binary, ksize=3, is_jpg=True):
    if not is_jpg:
        ksize = ksize + 2
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize)
    )
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )
    
    return closed

def preprocess(img, ksize, is_jpg):
    
    # 1. 去块
    denoised = remove_jpeg_blocks(img, strength=3)
    # 2. 结构感知增强
    guided = guided_enhance(denoised, is_jpg=is_jpg)
    # 3. 边缘检测
    edges = cv2.Canny(guided, 50, 150)
    # 4. 连接断线
    connected = connect_broken_edges(edges, ksize, is_jpg)
    return connected
