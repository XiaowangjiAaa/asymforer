import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2


class MultiScaleSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 确保中间通道数至少为1
        mid_channels = max(in_channels // reduction, 1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        channel_weights = self.conv(avg_pool + max_pool)
        return channel_weights * x  # 通道注意力加权

class SpatialFrequencyAttention(nn.Module):
    """频域增强的空间注意力 (修复尺寸不匹配问题)"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(in_channels, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # 频域特征提取
        x_fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(x_fft)  # [B, C, H, W//2+1]

        # 空间注意力生成
        spatial_att = self.conv_spatial(x)  # [B, 1, H, W]

        # 频域特征尺寸适配
        magnitude_avg = magnitude.mean(dim=1, keepdim=True)  # [B, 1, H, W//2+1]
        magnitude_avg = F.interpolate(
            magnitude_avg,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, H, W]

        # 特征融合
        fused_att = spatial_att + magnitude_avg
        return self.sigmoid(fused_att)
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class RepVggBlock(nn.Module):
    """结构重参数化卷积块 (参考RT-DETR的CCFF模块[1](@ref))"""

    def __init__(self, ch_in, ch_out, act='silu'):
        super().__init__()
        self.conv3x3 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv1x1 = ConvNormLayer(ch_in, ch_out, 1, 1, act=None)
        self.act = nn.SiLU() if act == 'silu' else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv3x3(x) + self.conv1x1(x))


class CCFF(nn.Module):
    """跨尺度上下文特征融合模块（完整修复版）"""

    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        self.in_channels = in_channels  # 输入拼接后的通道数 (512)

        # 通道注意力（输入为拼接后的512通道）
        self.channel_att = MultiScaleSEBlock(in_channels)

        # 空间注意力（输入为拼接后的512通道）
        self.spatial_att = SpatialFrequencyAttention(in_channels)

        # 特征融合块（使用多个RepVggBlock）
        self.fusion_blocks = nn.Sequential(
            *[RepVggBlock(in_channels, in_channels) for _ in range(num_blocks)]
        )

        # 输出适配层
        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, depth_feat, rgb_feat):
        """
        前向传播流程：
        1. 拼接深度和RGB特征
        2. 应用通道注意力
        3. 应用空间注意力
        4. 融合注意力权重
        5. 通过重参数化卷积块融合特征
        6. 输出适配
        """
        # 1. 拼接特征 [B, 512, H, W]
        fused = torch.cat([depth_feat, rgb_feat], dim=1)

        # 2. 通道注意力 [B, 512, 1, 1]
        channel_weights = self.channel_att(fused)

        # 3. 空间注意力 [B, 1, H, W]
        spatial_weights = self.spatial_att(fused)

        # 4. 融合注意力权重 [B, 512, H, W]
        # 通道权重广播到空间尺寸 + 空间权重广播到通道维度
        att_weights = fused*channel_weights * spatial_weights

        # 5. 应用注意力并融合特征
        #att_fused = fused * att_weights
        fused_out = self.fusion_blocks(att_weights)

        # 6. 输出适配
        return self.output_conv(fused_out)



class DoubleConv(nn.Module):
    """双卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """标准UNet模型"""

    def __init__(self, num_classes=40):
        super().__init__()
        # Encoder
        self.left_conv_1 = DoubleConv(3, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 128)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(128, 256)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(256, 512)
        self.down_4 = nn.MaxPool2d(2, 2)

        # Center
        self.center_conv = DoubleConv(512, 1024)

        # Decoder
        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)

        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)

        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)

        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # Output
        self.output = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.left_conv_1(x)
        x1d = self.down_1(x1)

        x2 = self.left_conv_2(x1d)
        x2d = self.down_2(x2)

        x3 = self.left_conv_3(x2d)
        x3d = self.down_3(x3)

        x4 = self.left_conv_4(x3d)
        x4d = self.down_4(x4)

        # Center
        x5 = self.center_conv(x4d)

        # Decoder
        x6u = self.up_1(x5)
        x6 = self.right_conv_1(torch.cat([x6u, x4], 1))

        x7u = self.up_2(x6)
        x7 = self.right_conv_2(torch.cat([x7u, x3], 1))

        x8u = self.up_3(x7)
        x8 = self.right_conv_3(torch.cat([x8u, x2], 1))

        x9u = self.up_4(x8)
        x9 = self.right_conv_4(torch.cat([x9u, x1], 1))

        return self.output(x9)


class B0_T(nn.Module):
    """RGB-D融合网络"""

    def __init__(self, num_classes):
        super().__init__()
        # 关键修复：正确传入通道参数
        self.fusion = CCFF(
            in_channels=4,  # RGB(3) + Depth(1) = 4通道
            out_channels=3  # 输出3通道作为UNet输入
        )
        self.unet = UNet(num_classes)

    def forward(self, rgb, depth):
        # 深度图通道对齐 (1ch→1ch)
        if depth.shape[1] != 1:
            depth = depth[:, :1]  # 取第一通道

        # 特征融合 (4→3通道)
        fused = self.fusion(depth, rgb)

        # 通过UNet处理
        return self.unet(fused)


if __name__ == '__main__':
    # 创建修复后的模型
    model = B0_T(num_classes=40)
    model.eval()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 测试输入（RGB:3通道，深度图:1通道）
    rgb = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)  # 确保深度图为单通道

    # 前向测试
    with torch.no_grad():
        fused = model.fusion(depth, rgb)
        output = model(rgb, depth)

        print(f"输入: RGB={rgb.shape}, Depth={depth.shape}")
        print(f"融合后: {fused.shape} (应变为3通道)")
        print(f"UNet输出: {output.shape}")

    # 性能测试
    import time

    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = model(rgb, depth)
        print(f"平均推理时间: {(time.time() - start) / 10:.4f}s")