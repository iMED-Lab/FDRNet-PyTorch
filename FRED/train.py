import os
import torch
import torchvision.models as models
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import matplotlib.pyplot as plt
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x):
        t = rfft2(x, dim = (-2,-1), norm='backward')
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH
class PerceptualLoss(torch.nn.Module):
    def __init__(self, layers=[0,12,34], device=None):
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)
        self.l1 = torch.nn.L1Loss(reduction='mean')
        self.device=device

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        input_features = self.get_features(input)
        target_features = self.get_features(target)
        input_features = list(input_features.values())
        target_features = list(target_features.values())
        loss = 0
        for layer in range(len(input_features)):
            input_feature = input_features[layer]
            #print(input_feature)
            target_feature = target_features[layer]
            loss += self.l1(input_feature,target_feature)
            #loss += torch.mean((input_feature-target_feature) ** 2)
        return loss

    def get_features(self,x):
        features = {}
        for i, module in enumerate(self.vgg):
            #print(i,module)
            x = module(x)
            if i in self.layers:
                features[i] = x
        return features

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    perceptual_loss = PerceptualLoss(device=device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    epoch_per_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_metrics=0

    # 用于保存每个 epoch 的损失值
    loss_history = {
        'context_loss': [],
        'fft_loss': [],
        'perceptual_loss': []
    }


    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)

            p_l1 = perceptual_loss(pred_img[0].to(device), label_img4.to(device))
            p_l2 = perceptual_loss(pred_img[1].to(device), label_img2.to(device))
            p_l3 = perceptual_loss(pred_img[2].to(device), label_img.to(device))
            loss_content = l1+l2+l3
            loss_per = p_l1 + p_l2 + p_l3


            #基于fft的loss
            label_fft1 = rfft(label_img4)    #新版本torch已经删除了rfft,此处改写
            pred_fft1 = rfft(pred_img[0])
            label_fft2 = rfft(label_img2)
            pred_fft2 = rfft(pred_img[1])
            label_fft3 = rfft(label_img)
            pred_fft3 = rfft(pred_img[2])

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss = loss_content + 0.1 * loss_fft + 0.01 * loss_per
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_per_adder(loss_per.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()

        # 保存每个 epoch 的损失值
        loss_history['context_loss'].append(epoch_pixel_adder.average())
        loss_history['fft_loss'].append(epoch_fft_adder.average())
        loss_history['perceptual_loss'].append(epoch_per_adder.average())

        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_per_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val_uwa = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average uwa_metrics %.2f ' % (epoch_idx, val_uwa))
            writer.add_scalar('uwa_metrics', val_uwa, epoch_idx)
            if val_uwa >= best_metrics:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))

        torch.cuda.empty_cache()
        # 绘制损失曲线并保存
        _plot_loss_curves(loss_history, save_dir=args.model_save_dir)

    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)


def _plot_loss_curves(loss_history, save_dir):
    """
    绘制多个损失曲线并保存到指定文件夹
    """
    # 创建保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history['context_loss']) + 1)

    plt.plot(epochs, loss_history['context_loss'], label='Context Loss')
    plt.plot(epochs, loss_history['fft_loss'], label='FFT Loss')
    plt.plot(epochs, loss_history['perceptual_loss'], label='perceptual Loss')

    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图表
    save_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Loss curves saved to {save_path}")