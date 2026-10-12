import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

IMG_SIZE = 384

# 定义输入目录与输出新目录的映射
DIR_MAPPING = {
    "/root/shared-nvme/xiaolunwen/dataset/archive/train_images": "/root/shared-nvme/xiaolunwen/dataset/archive/train_images_processed",
    "/root/shared-nvme/xiaolunwen/dataset/archive/val_images": "/root/shared-nvme/xiaolunwen/dataset/archive/val_images_processed",
    "/root/shared-nvme/xiaolunwen/dataset/archive/test_images": "/root/shared-nvme/xiaolunwen/dataset/archive/test_images_processed"
}

def crop_and_pad_to_square(img, tol=7):
    if img.ndim == 2: mask = img > tol
    else: mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > tol
    if not mask.any(): return img
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    cropped = img[row_start:row_end, col_start:col_end]
    h, w = cropped.shape[:2]
    max_side = max(h, w)
    top = (max_side - h) // 2
    bottom = max_side - h - top
    left = (max_side - w) // 2
    right = max_side - w - left
    return cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_single_image(args):
    in_path, out_path = args
    image = cv2.imread(in_path)
    if image is None: return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_and_pad_to_square(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    
    # 统一存为 JPEG 格式以优化读取速度
    Image.fromarray(image).save(out_path, quality=95)

if __name__ == '__main__':
    for in_dir, out_dir in DIR_MAPPING.items():
        print(f"Processing directory: {in_dir} -> {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        
        tasks = []
        for f in os.listdir(in_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                in_path = os.path.join(in_dir, f)
                # 强制替换后缀为 .jpg
                out_filename = os.path.splitext(f)[0] + '.jpg'
                out_path = os.path.join(out_dir, out_filename)
                tasks.append((in_path, out_path))
                
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))