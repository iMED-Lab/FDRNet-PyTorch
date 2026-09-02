#!/usr/bin/env python3
# crop_to_even.py
import os
from PIL import Image

ROOT_DIR = '../ResNet-pytorch-main/hyper-kvasir/original'          # 改成你的根目录
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def crop_to_even(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(EXTS):
                path = os.path.join(dirpath, fname)
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        # 计算最接近原尺寸的 2 的倍数
                        new_w = (w // 2) * 2
                        new_h = (h // 2) * 2
                        if new_w == w and new_h == h:
                            continue  # 已是 2 的倍数
                        # 居中裁剪
                        left   = (w - new_w) // 2
                        upper  = (h - new_h) // 2
                        right  = left + new_w
                        lower  = upper + new_h
                        cropped = img.crop((left, upper, right, lower))
                        cropped.save(path)
                        print(f"{path} 已裁剪为 {new_w}×{new_h}")
                except Exception as e:
                    print(f"处理失败：{path} ({e})")

if __name__ == '__main__':
    crop_to_even(ROOT_DIR)