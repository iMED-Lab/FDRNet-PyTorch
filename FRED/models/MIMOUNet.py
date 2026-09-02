import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16,r=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
        )
        self.fusion = ConvBlock(inchannel, inchannel, 1,1,0,bias=True)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)
        self.local_att = nn.Sequential(torch.nn.Conv2d(inchannel,inchannel//r,kernel_size=1,stride=1,padding=0),
                                        nn.BatchNorm2d(inchannel//r),
                                        nn.ReLU(inplace=True),
                                       torch.nn.Conv2d(inchannel//r,inchannel,kernel_size=1,stride=1,padding=0),
                                       nn.BatchNorm2d(inchannel))
        self.S = nn.Sigmoid()

    def forward(self, x1,x2,x4):
        x = torch.cat([x1, x2, x4], dim=1)
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        x_l = self.local_att(x)
        fusion = self.fusion(self.S(avg+max))
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x_l
        fusion = self.outlayer(fusion)
        return fusion

class EnhancedFusionLayer(nn.Module):
    def __init__(self, inchannels, outchannels, reduction=16, r=4, groups=4):
        super(EnhancedFusionLayer, self).__init__()

        # 多尺度特征提取
        self.multi_scale = nn.ModuleDict({
            'conv1x1': nn.Conv2d(inchannels, inchannels // 2, 1),
            'conv3x3': nn.Conv2d(inchannels, inchannels // 2, 3, padding=1, groups=groups),
            'conv5x5': nn.Conv2d(inchannels, inchannels // 2, 5, padding=2, groups=groups)
        })
        self.scale_fusion = nn.Conv2d(inchannels //2 * 3, inchannels, 1)

        # 混合注意力机制
        self.channel_att = ChannelAttention(inchannels, inchannels, reduction)
        self.spatial_att = SpatialAttention(r)

        # 交叉特征交互
        self.cross_gate = CrossFeatureGate(inchannels)

        # 输出转换
        self.out_conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        # 多尺度特征融合
        merged = torch.cat([x1, x2, x3], dim=1)

        # 多尺度处理
        scale_feats = []
        for name, conv in self.multi_scale.items():
            scale_feats.append(conv(merged))
        scale_feats = torch.cat(scale_feats, dim=1)
        scale_feats = self.scale_fusion(scale_feats)

        # 混合注意力
        ca = self.channel_att(scale_feats)
        sa = self.spatial_att(scale_feats)
        attended = ca * sa

        # 交叉门控增强
        enhanced = self.cross_gate(attended, merged)

        # 输出转换
        return self.out_conv(enhanced)

class ChannelAttention(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16,r=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
        )
        self.fusion = ConvBlock(inchannel, inchannel, 1,1,0,bias=True)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)
        self.local_att = nn.Sequential(torch.nn.Conv2d(inchannel,inchannel//r,kernel_size=1,stride=1,padding=0),
                                        nn.BatchNorm2d(inchannel//r),
                                        nn.ReLU(inplace=True),
                                       torch.nn.Conv2d(inchannel//r,inchannel,kernel_size=1,stride=1,padding=0),
                                       nn.BatchNorm2d(inchannel))
        self.S = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        x_l = self.local_att(x)
        fusion = self.fusion(self.S(avg+max))
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x_l
        fusion = self.outlayer(fusion)
        return fusion


class SpatialAttention(nn.Module):
    """高效空间注意力"""
    def __init__(self, r=4):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg, max], dim=1)
        return self.conv(feat)


class CrossFeatureGate(nn.Module):
    """交叉特征门控"""

    def __init__(self, channels):
        super(CrossFeatureGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, att_feat, raw_feat):
        combined = torch.cat([att_feat, raw_feat], dim=1)
        gate = self.gate(combined)
        return att_feat * gate + raw_feat * (1 - gate)

class Frequency_separation(torch.nn.Module):
    def __init__(self):
        super(Frequency_separation, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x_l1 = self.avgpool(x)
        x_l = F.interpolate(x_l1, scale_factor=2, mode='bilinear', align_corners=True)
        x_h = x - x_l
        return x_h,x_l

class MBB(nn.Module):
    def __init__(self, n_feats):
        super(MBB, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),
                                 nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),
                                 nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


#原始的频域融合模块
class Frequency_fusion(torch.nn.Module):
    def __init__(self,inchannel):
        super(Frequency_fusion, self).__init__()
        self.mbb = MBB(inchannel)
        self.ca = CALayer(2 * inchannel)
        self.conv = nn.Sequential(nn.Conv2d(2 * inchannel, inchannel, 3, 1, 1, bias=False),
                                  nn.Conv2d(inchannel, inchannel, 1, 1, 0, bias=False))

    def forward(self,x_l,x_h):
        x_h = self.mbb(x_h)
        x_l = self.mbb(self.mbb(self.mbb(x_l)))
        x = torch.cat([x_h, x_l], dim=1)
        x = self.ca(x)
        x = self.conv(x)

        return x

class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        #添加频域残差块
        BasicConv = BasicConv_do
        ResBlock = enhancedResBlock

        self.fre_sep = Frequency_separation()
        self.Encoder_h = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock),
            EBlock(base_channel*2, num_res, ResBlock),
            EBlock(base_channel*4, num_res, ResBlock),
        ])

        self.feat_extract_h = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder_h = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs_h = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut_h = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs_h = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1_h = FAM(base_channel * 4)
        self.SCM1_h = SCM(base_channel * 4)
        self.FAM2_h = FAM(base_channel * 2)
        self.SCM2_h = SCM(base_channel * 2)
        self.fusion2_h = EnhancedFusionLayer(base_channel * 7, base_channel * 2)
        self.fusion1_h = EnhancedFusionLayer(base_channel * 7, base_channel)

        self.drop1_h = nn.Dropout2d(0.1)
        self.drop2_h = nn.Dropout2d(0.1)

        self.Encoder_l = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock),
            EBlock(base_channel * 2, num_res, ResBlock),
            EBlock(base_channel * 4, num_res, ResBlock),
        ])

        self.feat_extract_l = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder_l = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs_l = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut_l = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs_l = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.FAM1_l = FAM(base_channel * 4)
        self.SCM1_l = SCM(base_channel * 4)
        self.FAM2_l = FAM(base_channel * 2)
        self.SCM2_l = SCM(base_channel * 2)
        self.fusion2_l = EnhancedFusionLayer(base_channel * 7, base_channel * 2)
        self.fusion1_l = EnhancedFusionLayer(base_channel * 7, base_channel)

        self.drop1_l = nn.Dropout2d(0.1)
        self.drop2_l = nn.Dropout2d(0.1)

        self.fre_fus1 = Frequency_fusion(base_channel)
        self.fre_fus2 = Frequency_fusion(base_channel * 2)
        self.fre_fus3 = Frequency_fusion(base_channel * 4)

    def forward(self, x):
        x_2 = F.interpolate(x,scale_factor = 0.5)
        x_4 = F.interpolate(x_2,scale_factor = 0.5)
        x_h,x_l = self.fre_sep(x)
        xh_2 = F.interpolate(x_h, scale_factor=0.5)
        xh_4 = F.interpolate(xh_2, scale_factor=0.5)
        zh2 = self.SCM2_h(xh_2)
        zh4 = self.SCM1_h(xh_4)

        xl_2 = F.interpolate(x_l, scale_factor=0.5)
        xl_4 = F.interpolate(xl_2, scale_factor=0.5)
        zl2 = self.SCM2_l(xl_2)
        zl4 = self.SCM1_l(xl_4)

        outputs = list()

        xh_ = self.feat_extract_h[0](x_h)
        resh1 = self.Encoder_h[0](xh_)
        xl_ = self.feat_extract_l[0](x_l)
        resl1 = self.Encoder_l[0](xl_)

        z_h = self.feat_extract_h[1](resh1)
        z_h = self.FAM2_h(z_h, zh2)
        resh2 = self.Encoder_h[1](z_h)
        z_l = self.feat_extract_l[1](resl1)
        z_l = self.FAM2_l(z_l, zl2)
        resl2 = self.Encoder_l[1](z_l)

        z_h = self.feat_extract_h[2](resh2)
        z_h = self.FAM1_h(z_h, zh4)
        z_h = self.Encoder_h[2](z_h)
        z_l = self.feat_extract_l[2](resl2)
        z_l = self.FAM1_l(z_l, zl4)
        z_l = self.Encoder_l[2](z_l)

        zh12 = F.interpolate(resh1, scale_factor=0.5)
        zh21 = F.interpolate(resh2, scale_factor=2)
        zh42 = F.interpolate(z_h, scale_factor=2)
        zh41 = F.interpolate(zh42, scale_factor=2)
        zl12 = F.interpolate(resl1, scale_factor=0.5)
        zl21 = F.interpolate(resl2, scale_factor=2)
        zl42 = F.interpolate(z_l, scale_factor=2)
        zl41 = F.interpolate(zl42, scale_factor=2)

        resh2 = self.fusion2_h(zh12, resh2, zh42)
        resh1 = self.fusion1_h(resh1, zh21, zh41)
        resl2 = self.fusion2_l(zl12, resl2, zl42)
        resl1 = self.fusion1_l(resl1, zl21, zl41)

        resh2 = self.drop2_h(resh2)
        resh1 = self.drop1_h(resh1)
        resl2 = self.drop2_l(resl2)
        resl1 = self.drop1_l(resl1)

        z_h = self.Decoder_h[0](z_h)
        z_l = self.Decoder_l[0](z_l)
        out = self.fre_fus3(z_l, z_h)
        out = self.ConvsOut_l[0](out+z_l+z_h)
        z_h = self.feat_extract_h[3](z_h)
        z_l = self.feat_extract_l[3](z_l)
        outputs.append(out + x_4)

        z_h = torch.cat([z_h, resh2], dim=1)
        #z_h = self.fusion2_h(z_h)
        z_h = self.Convs_h[0](z_h)
        z_h = self.Decoder_h[1](z_h)
        z_l = torch.cat([z_l, resl2], dim=1)
        #z_l = self.fusion2_l(z_l)
        z_l = self.Convs_l[0](z_l)
        z_l = self.Decoder_l[1](z_l)
        out = self.fre_fus2(z_l,z_h)
        out = self.ConvsOut_l[1](out+z_l+z_h)
        z_h = self.feat_extract_h[4](z_h)
        z_l = self.feat_extract_l[4](z_l)
        outputs.append(out+x_2)

        z_h = torch.cat([z_h, resh1], dim=1)
        #z_h = self.fusion1_h(z_h)
        z_h = self.Convs_h[1](z_h)
        z_h = self.Decoder_h[2](z_h)
        z_l = torch.cat([z_l, resl1], dim=1)
        #z_l = self.fusion1_l(z_l)
        z_l = self.Convs_l[1](z_l)
        z_l = self.Decoder_l[2](z_l)
        out = self.fre_fus1(z_l,z_h)
        out = self.feat_extract_l[5](out+z_l+z_h)
        outputs.append(out+x)


        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')
