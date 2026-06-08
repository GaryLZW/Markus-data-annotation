def save_for_ocr(img, mask, out_path,
                 invert=True, pad=10):
    """
    img: 原图（BGR）
    mask: 结构式掩膜（0/255）
    """
    # 1. 裁剪 ROI
    x, y, w, h = cv2.boundingRect(mask)
    roi = mask[y:y+h, x:x+w]
    # 2. 可选反色
    if invert:
        roi = cv2.bitwise_not(roi)
    # 3. 加 padding
    roi = cv2.copyMakeBorder(
        roi, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=255 if invert else 0
    )
    cv2.imwrite(out_path, roi)
