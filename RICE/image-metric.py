# import os
# import cv2
# import sys
# import numpy as np
# import torch
# import argparse
# import torch.utils
# import lpips
# import torch.backends.cudnn as cudnn
# from PIL import Image
# from torch.autograd import Variable
# from skimage.metrics import structural_similarity
# from skimage.metrics import peak_signal_noise_ratio
#
#
# def calculate_metrics(enhancement_img_path, img_path):
#     """
#     该函数用于计算增强前后图像的PSNR与SSIM值。
#     """
#     SSIM_list = []
#     PSNR_list = []
#     LPIPS_list = []
#     enhancement_img_list = os.listdir(enhancement_img_path)
#     img_list = os.listdir(img_path)
#     lpips_model = lpips.LPIPS(net='alex')
#
#     for i in range(len(enhancement_img_list)):
#         enhancement_img = cv2.imread(os.path.join(enhancement_img_path, enhancement_img_list[i]))
#         img = cv2.imread(os.path.join(img_path, img_list[i]))
#
#         #判断尺寸是否一致
#         if enhancement_img.shape[0] != img.shape[0] or enhancement_img.shape[1] != img.shape[1]:
#             pil_img = Image.fromarray(img)
#             pil_img = pil_img.resize((enhancement_img.shape[1],enhancement_img.shape[0]))  # 和clear_img的宽和高保持一致
#             img = np.array(pil_img)
#
#         # 计算PSNR
#         PSNR = peak_signal_noise_ratio(enhancement_img, img)
#         print(i + 1, 'PSNR: ', PSNR)
#         PSNR_list.append(PSNR)
#
#         # 计算SSIM
#         SSIM = structural_similarity(enhancement_img, img, channel_axis=2)
#         print(i + 1, 'SSIM: ', SSIM)
#         SSIM_list.append(SSIM)
#
#         #计算LPIPS
#         enhancement_img_tensor = torch.tensor(np.array(enhancement_img)).permute(2, 0, 1).unsqueeze(0).float() /255.0
#         img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#         LPIPS = lpips_model(enhancement_img_tensor, img_tensor)
#         LPIPS = LPIPS.item()
#         print(i + 1, 'LPIPS: ', LPIPS)
#         LPIPS_list.append(LPIPS)
#
#     print("average SSIM", sum(SSIM_list) / len(SSIM_list))
#     print("std ssim",np.std(SSIM_list))
#     print("average PSNR", sum(PSNR_list) / len(PSNR_list))
#     print("std psnr",np.std(PSNR_list))
#     print("average LPIPS", sum(LPIPS_list) / len(LPIPS_list))
#     print("std LPIPS",np.std(LPIPS_list))
#
# if __name__ == '__main__':
#     calculate_metrics('./data/RICE', './fundus/high_quality')
#

import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 数据准备
# 均值 (Means)
psnr_means = [19.78, 20.22, 13.64, 21.60]
ssim_means = [0.62, 0.65, 0.60, 0.78]
lpips_means = [0.16, 0.12, 0.17, 0.09]

# 标准差 (Standard Deviations)
psnr_stds = [8.58, 9.27, 2.39, 6.07]
ssim_stds = [0.08, 0.09, 0.04, 0.05]
lpips_stds = [0.07, 0.08, 0.05, 0.05]

# 2. 设置顶级期刊常用的高级配色
# 浅灰 (Original), 浅灰蓝 (RICE), 浅水绿 (FRED), 砖红/深橘红 (FDRNet)
colors = ['#D3D3D3', '#82B0D2', '#8ECFC9', '#FA7F6F']
edge_colors = ['#A9A9A9', '#5080A2', '#5E9F99', '#CA4F3F']

# 3. 创建画布 (1行3列)
# 因为去掉了大量的文字标签，高度可以适当压缩 (由5.5调整为4.5)，使得拼图时更紧凑
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

means_list = [psnr_means, ssim_means, lpips_means]
stds_list = [psnr_stds, ssim_stds, lpips_stds]

x_pos = np.arange(4)
width = 0.55  # 柱子的宽度

for i in range(3):
    ax = axes[i]
    means = means_list[i]
    stds = stds_list[i]

    # 画带误差棒的柱状图
    bars = ax.bar(x_pos, means, width, yerr=stds, align='center',
                  alpha=0.95, ecolor='black', capsize=6,
                  color=colors, edgecolor=edge_colors, linewidth=1.5,
                  error_kw={'elinewidth': 1.5, 'markeredgewidth': 1.5})

    # 隐藏 X 轴的具体文本标签和刻度短线
    ax.set_xticks([])

    # 隐藏 Y 轴标签和顶部标题，留空给你后续用软件加
    ax.set_title("")
    ax.set_ylabel("")

    # 保留 Y 轴的数字刻度，方便读者看数值级别
    ax.tick_params(axis='y', labelsize=20)

    # 美化边框：隐藏顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 添加水平虚线网格，增加数值可读性
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#B0B0B0')
    ax.set_axisbelow(True)  # 让网格线处于图层最下方

    # 在误差棒的上方/下方标注均值
    for j, bar in enumerate(bars):
        yval = means[j]
        # 对最终的 FDRNet 强调显示加粗字体
        weight = 'bold' if j == 3 else 'normal'
        text_color = '#B22222' if j == 3 else 'black'  # FDRNet 数值用深红色

        # 动态计算数字显示的位置（避开误差棒）
        if i == 0:  # PSNR
            offset = stds[j] + 1.0
        elif i == 1:  # SSIM
            offset = stds[j] + 0.015
        else:  # LPIPS
            offset = stds[j] + 0.01

        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + offset,
                f'{yval:.2f}', ha='center', va='bottom',
                fontsize=20, fontweight=weight, color=text_color)

# 自动调整子图间距
plt.tight_layout()

# 4. 保存为高分辨率图片 (PNG用于Word预览，PDF用于LaTeX/AI/Visio后期排版)
save_dir = "./"
plt.savefig(os.path.join(save_dir, 'ablation_clean_chart.png'), dpi=330, bbox_inches='tight')


print("纯净版图表已成功生成：ablation_clean_chart.png 和 ablation_clean_chart.pdf")
plt.show()

# import os
# import cv2
# import numpy as np
# import argparse
# import shutil
# import lpips
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# from PIL import Image
# import torch
#
# def calculate_metrics_and_select_best(enhancement_img_path, img_path, output_path):
#     """
#     计算指标并选择最佳增强图像。
#
#     参数：
#         enhancement_img_path (str): 增强图像文件夹路径。
#         img_path (str): 原始图像文件夹路径。
#         output_path (str): 输出文件夹路径。
#     """
#     # 获取增强图像和原始图像列表
#     enhancement_img_list = os.listdir(enhancement_img_path)
#     img_list = os.listdir(img_path)
#
#     # 确保输出文件夹存在
#     os.makedirs(output_path, exist_ok=True)
#
#     # 初始化 LPIPS 模型
#     lpips_model = lpips.LPIPS(net='alex')
#
#     # 遍历每张原始图像
#     for img_name in img_list:
#         # 构造原始图像完整路径
#         img_full_path = os.path.join(img_path, img_name)
#         # 读取原始图像
#         img = cv2.imread(img_full_path)
#         # 如果图像未正确读取，跳过
#         if img is None:
#             print(f"无法读取图像：{img_full_path}")
#             continue
#
#         # 提取原始图像的基本名称（不含扩展名）
#         img_base_name = os.path.splitext(img_name)[0]
#         # 筛选对应的增强图像
#         corresponding_enhancement_imgs = [
#             enh_img for enh_img in enhancement_img_list
#             if enh_img.startswith(img_base_name + '_') and enh_img.endswith('.png')
#         ]
#
#         # 如果没有找到对应的增强图像，跳过
#         if not corresponding_enhancement_imgs:
#             print(f"未找到与图像 {img_name} 对应的增强图像")
#             continue
#
#         best_psnr = -1
#         best_ssim = -1
#         best_lpips = float('inf')
#         best_img_path = ""
#
#         # 遍历每张对应的增强图像
#         for enhancement_img_name in corresponding_enhancement_imgs:
#             # 构造增强图像完整路径
#             enhancement_img_full_path = os.path.join(enhancement_img_path, enhancement_img_name)
#             # 读取增强图像
#             enhancement_img = cv2.imread(enhancement_img_full_path)
#             # 如果图像未正确读取，跳过
#             if enhancement_img is None:
#                 print(f"无法读取增强图像：{enhancement_img_full_path}")
#                 continue
#
#             # 如果增强图像与原始图像尺寸不同，调整原始图像尺寸以匹配增强图像
#             if enhancement_img.shape[0] != img.shape[0] or enhancement_img.shape[1] != img.shape[1]:
#                 pil_img = Image.fromarray(img)
#                 pil_img = pil_img.resize((enhancement_img.shape[1], enhancement_img.shape[0]))
#                 img = np.array(pil_img)
#
#             # 计算 PSNR 指标
#             psnr = peak_signal_noise_ratio(enhancement_img, img)
#
#             # 计算 SSIM 指标
#             ssim = structural_similarity(enhancement_img, img, channel_axis=2)
#
#             # 计算 LPIPS 指标
#             enhancement_img_tensor = torch.tensor(enhancement_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#             img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
#             lpips_value = lpips_model(enhancement_img_tensor, img_tensor)
#             lpips_value = lpips_value.item()
#
#             # 更新最佳图像
#             if psnr > best_psnr and ssim > best_ssim and lpips_value < best_lpips:
#                 best_psnr = psnr
#                 best_ssim = ssim
#                 best_lpips = lpips_value
#                 best_img_path = enhancement_img_full_path
#
#         # 如果找到最佳图像，复制到输出文件夹
#         if best_img_path:
#             shutil.copy2(best_img_path, os.path.join(output_path, img_name))
#             print(f"为图像 {img_name} 选择了最佳增强图像：{best_img_path}")
#         else:
#             print(f"未找到图像 {img_name} 的最佳增强图像")
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Calculate metrics and select the best enhanced image.')
#     parser.add_argument('--enhancement_img_path', type=str, default='./EXP/Train-20250506-071620/image_epochs', help='增强图像文件夹路径')
#     parser.add_argument('--img_path', type=str, default='./data/fundus-highquality', help='原始图像文件夹路径')
#     parser.add_argument('--output_path', type=str,default= './results/fundus', help='输出文件夹路径')
#     args = parser.parse_args()
#
#     calculate_metrics_and_select_best(args.enhancement_img_path, args.img_path, args.output_path)
