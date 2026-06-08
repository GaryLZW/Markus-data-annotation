import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from image_processor.process import process_one_image  # 你已有的处理函数
RAW_DIR = "../ocr-img-test/raw"
OUT_DIR = "../ocr-img-test/processed"
os.makedirs(OUT_DIR, exist_ok=True)
def worker(img_name):
    try:
        in_path = os.path.join(RAW_DIR, img_name)
        out_path = os.path.join(OUT_DIR, f"{img_name[:8]}.png")
        process_one_image(in_path, out_path)
        return f":white_check_mark: {img_name}"
    except Exception as e:
        return f":x: {img_name}: {e}"
def main():
    img_names = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(img_names)} images")
    # CPU 核心数（留 1~2 核给系统）
    #max_workers = os.cpu_count() - 1 or 1
    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, name) for name in img_names]
        for future in as_completed(futures):
            print(future.result())
if __name__ == "__main__":
    main()
