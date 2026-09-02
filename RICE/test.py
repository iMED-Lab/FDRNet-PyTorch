'''
import os
import cv2
import sys
import numpy as np
import torch
import argparse
import torch.utils
import lpips
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import tensorflow as tf
from scipy.stats import entropy
from multi_read_data import MemoryFriendlyLoader
import torchvision.models as models
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/endoscopy-fredwoDDCACI',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/woloss-endo', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./EXP/Train-20260325-092330/model_epochs/weights_777.pt', help='location of the data corpus') #./EXP/endoscopy-3.1.2.5/model_epochs/weights_860.pt   uwa-3.1.2.5  endoscopy-crop 502.pt  cfp-800
parser.add_argument('--gpu', type=str, default='3', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('/')[-1].split('.')[0]
            i, r = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            ui_path = save_path + '/' + 'illu'
            if not os.path.exists(ui_path):
                os.makedirs(ui_path)
            ui_path = ui_path + '/' + u_name
            save_images(r, u_path)
            save_images(i, ui_path)



if __name__ == '__main__':
    main()
'''

import os
import cv2
import sys
import numpy as np
import torch
import argparse
import torch.utils
import lpips
import time  # 新增：用于计算推理时间
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import tensorflow as tf
from scipy.stats import entropy
from multi_read_data import MemoryFriendlyLoader
import torchvision.models as models
from thop import profile
from thop import clever_format
import copy
parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/endoscopy-fredwoDDCACI',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/woloss-endo', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./EXP/Train-20260325-092330/model_epochs/weights_777.pt',
                    help='location of the data corpus')
parser.add_argument('--gpu', type=str, default='1', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model)
    model = model.cuda()
    model.eval()

    # ==========================================
    # 新增模块：计算 Params, FLOPs 和 推理时间
    # ==========================================
    print("\n" + "=" * 40)
    print("Evaluating Model Efficiency...")
    # 构造虚拟输入张量 (Batch_Size=1, Channels=3, H=1024, W=1024)
    dummy_input = torch.randn(1, 3, 1024, 1024).cuda()
    model_for_time = copy.deepcopy(model)

    with torch.no_grad():
        # 1. 计算 FLOPs 和 Params
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
        print(f"FLOPs (MACs): {macs_str}")
        print(f"Parameters: {params_str}")

        del model
        torch.cuda.empty_cache()

        # 2. GPU 预热 (消除初始化带来的耗时误差)
        print("Starting GPU warm-up...")
        for _ in range(50):
            _ = model_for_time(dummy_input)

        # 3. 测量推理时间
        iterations = 100
        times = []
        print("Measuring inference time...")
        for _ in range(iterations):
            torch.cuda.synchronize()  # 等待之前的计算完成
            starter = time.time()

            _ = model_for_time(dummy_input)

            torch.cuda.synchronize()  # 阻塞 CPU 直到本次 GPU 计算完成
            ender = time.time()
            times.append(ender - starter)

        mean_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        print(f"Input Shape: (1, 3, 1024, 1024)")
        print(f"Average Inference Time: {mean_time:.2f} ms (+/- {std_time:.2f} ms)")
        print(f"FPS: {1000 / mean_time:.2f}")
    print("=" * 40 + "\n")
    # ==========================================

    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            # 移除了废弃的 Variable(volatile=True) 写法
            input = input.cuda()
            image_name = image_name[0].split('/')[-1].split('.')[0]
            i, r = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            ui_path = save_path + '/' + 'illu'
            if not os.path.exists(ui_path):
                os.makedirs(ui_path)
            ui_path = ui_path + '/' + u_name
            save_images(r, u_path)
            save_images(i, ui_path)


if __name__ == '__main__':
    main()
    # calculate_metrics(args.save_path, args.data_path)
