# import os
# import torch
# from torchvision.transforms import functional as F
# import numpy as np
# from utils import Adder
# from data import test_dataloader
# from skimage.metrics import peak_signal_noise_ratio
# import time
#
#
# def _eval(model, args):
#     state_dict = torch.load(args.test_model)
#     model.load_state_dict(state_dict['model'])
#     device = torch.device('cuda')#'cuda' if torch.cuda.is_available() else
#     dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
#     torch.cuda.empty_cache()
#     adder = Adder()
#     model.eval().to(device)
#     with torch.no_grad():
#         #psnr_adder = Adder()
#
#         # Hardware warm-up
#         for iter_idx, data in enumerate(dataloader):
#             input_img, label_img, _ = data
#             input_img = input_img.to(device)
#             tm = time.time()
#             _ = model(input_img)
#             _ = time.time() - tm
#
#             if iter_idx == 20:
#                 break
#
#         # Main Evaluation
#         for iter_idx, data in enumerate(dataloader):
#             input_img, label_img, name = data
#
#             input_img = input_img.to(device)
#
#             tm = time.time()
#
#             pred = model(input_img)[2]
#
#             elapsed = time.time() - tm
#             adder(elapsed)
#
#             pred_clip = torch.clamp(pred, 0, 1)
#
#             #pred_numpy = pred_clip.squeeze(0).cpu().numpy()
#             #label_numpy = label_img.squeeze(0).cpu().numpy()
#
#             if args.save_image:
#                 save_name = os.path.join(args.result_dir, name[0])
#                 save_name = save_name.replace("jpg","png")
#                 pred_clip += 0.5 / 255
#                 pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
#                 pred.save(save_name)

