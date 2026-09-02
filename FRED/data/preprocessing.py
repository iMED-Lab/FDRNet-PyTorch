
import os
from PIL import Image
import glob


def resize_image_to_4_divisible(image_path, output_path=None):
    """
    调整图片尺寸，使其宽度和高度都能被4整除，尽可能保持原始大小

    参数:
        image_path: 原始图片路径
        output_path: 处理后图片保存路径，默认为覆盖原始图片
    """
    # 如果未指定输出路径，则覆盖原图
    if output_path is None:
        output_path = image_path

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # 检查是否已经满足条件（长宽都能被4整除）
            if width % 4 == 0 and height % 4 == 0:
                return  # 无需处理

            # 计算新尺寸，确保都能被4整除
            # 思路：找到小于等于原尺寸且能被4整除的最大数
            new_width = width - (width % 4)
            new_height = height - (height % 4)

            # 特殊情况处理：如果调整后尺寸过小，适当放大以保证最小可读性
            if new_width < 4:
                new_width = 4
            if new_height < 4:
                new_height = 4

            # 调整图片大小，使用高质量的缩小算法
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 保存处理后的图片，保持原始格式
            resized_img.save(output_path)
            print(f"已调整: {image_path} -> {new_width}x{new_height}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")


def process_image_folder(folder_path):
    """处理指定文件夹中的所有图片"""
    # 支持的图片格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.gif', '*.tiff']
    image_files = []

    # 收集所有图片文件
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))

    if not image_files:
        print(f"在文件夹 {folder_path} 中未找到图片文件")
        return

    # 处理每张图片
    for img_file in image_files:
        resize_image_to_4_divisible(img_file)

    print(f"处理完成，共处理 {len(image_files)} 张图片")


if __name__ == "__main__":
    # 替换为你的图片文件夹路径
    target_folder = "../dataset/fundus/test/sharp"

    # 验证文件夹是否存在
    if not os.path.isdir(target_folder):
        print(f"错误: 文件夹 {target_folder} 不存在")
    else:
        process_image_folder(target_folder)

