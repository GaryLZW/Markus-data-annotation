from e_smile_workflow.ocsr_utils import needs_review, ocsr_predict, extract_and_judge_smiles
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# RAW_DIR = "../1_v2/pic"
# PROCESSED_DIR = "../1_v2/pic_processed"
# RAW_DIR = "../ocr-img-test/raw"
# PROCESSED_DIR = "../ocr-img-test/processed"
# OUT_DIR = "../ocr-img-test"
RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"
OUT_DIR = "dataset"


os.makedirs(OUT_DIR, exist_ok=True)

def worker(img_name):
    try:
        raw_img = os.path.join(RAW_DIR, img_name)
        cooked_img = os.path.join(PROCESSED_DIR, f"{img_name[:8]}.png")
        
        result = extract_and_judge_smiles(raw_img, cooked_img)
        
        return result
    except Exception as e:
        return f":x: {img_name}: {e}"


def main():
    output_csv = OUT_DIR + '/' +'smilestest.csv'

    img_names = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(img_names)} images")
    # CPU 核心数（留 1~2 核给系统）
    # max_workers = os.cpu_count() - 1 or 1
    max_workers = 2
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, name) for name in img_names]
        for future in as_completed(futures):
            print(future.result()["image"])

    for future in futures:
        results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()

















