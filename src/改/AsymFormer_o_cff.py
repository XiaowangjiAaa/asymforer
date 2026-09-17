from torch.nn import functional as F
from src.mix_transformer import OverlapPatchEmbed, mit_b0
from src.convnext import convnext_tiny
from thop import profile
from src.MLPDecoder import DecoderHead
import os


def load_pretrain2(net, pretrain_name):
    dir_path = os.getcwd()
    pretrain_path = os.path.join(dir_path, 'src/model_zoo/segformer/imagenet_pretrain', pretrain_name)
    print("Pretrain_path:", pretrain_path)
    net_dict = net.state_dict()

    pretrain_dict = torch.load(pretrain_path)

    dict = {k: v for k, v in pretrain_dict.items() if k in net_dict}
    net_dict.update(dict)
    net.load_state_dict(net_dict)
    return net


model1 = convnext_tiny(pretrained=True, drop_path_rate=0.3)
ft1 = model1.stages
stem = model1.downsample_layers
stem1 = [stem[0], stem[1], stem[2], stem[3]]
layers1 = [
    ft1[0],
    ft1[1],
    ft1[2],
    ft1[3]]

model2 = mit_b0()
# model2 = load_pretrain2(model2, pretrain_name='mit_b0.pth')
layers2 = [model2.block1, model2.block2, model2.block3, model2.block4]
stem2 = [model2.patch_embed1, model2.patch_embed2, model2.patch_embed3, model2.patch_embed4]
norm2 = [model2.norm1, model2.norm2, model2.norm3, model2.norm4]

import torch
from torch import nn


def channel_shuffle(x, groups: int):
    batchsize, N, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, N, groups, channels_per_group)

    # Transpose operation is not valid for 5D tensor, so we need to use permute
    x = x.permute(0, 1, 3, 2).contiguous()

    # flatten
    x = x.view(batchsize, N, -1)

    return x


class Cross_Atten_Lite_split(nn.Module):
    def __init__(self, inc1, inc2):
        super(Cross_Atten_Lite_split, self).__init__()
        self.midc1 = torch.tensor(inc1 // 4)
        self.midc2 = torch.tensor(inc2 // 4)

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout = nn.Dropout(0.2)
        self._init_weight()

    def forward(self, x, x1, x2):
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)

        x1 = self.bn_x1(x1)
        x2 = self.bn_x2(x2)

        kq1 = self.kq1(x1.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq2 = self.kq2(x2.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq = channel_shuffle(torch.cat([kq1, kq2], dim=2), 2)
        k1, q1, k2, q2 = torch.split(kq, self.midc2, dim=2)

        v = self.v_conv(x.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        v1, v2 = torch.split(v, self.midc1, dim=2)

        mat = torch.matmul(q1, k1.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v1 = torch.matmul(mat, v1)

        mat = torch.matmul(q2, k2.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v2 = torch.matmul(mat, v2)

        v = torch.cat([v1, v2], dim=2).view(batch_size, h, w, -1)
        v = self.out_conv(v)
        v = v.permute(0, 3, 1, 2)
        v = self.bn_last(v)
        v = v + x

        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

#Local Attention-Guided Feature Selection
class MultiScaleSEBlock(nn.Module):
    """多尺度通道注意力机制 (参考RT-DETR的通道混合器设计[1](@ref))"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        channel_weights = self.conv(avg_pool + max_pool)
        return x * channel_weights


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


class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class CCFF(nn.Module):
    """跨尺度上下文特征融合模块（简化输入接口）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 通道注意力（修复通道数保护）
        self.channel_att = MultiScaleSEBlock(in_channels)

        # 空间注意力
        self.spatial_att = SpatialFrequencyAttention(in_channels)

        # 空间特征融合模块
        self.fusion = nn.Sequential(
            IDSC(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            DSC(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # 通道降维层
        self.channel_reduce = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, fused):
        """只接收单个输入（已经拼接的特征）"""
        # 通道注意力 (修复：使用完整的通道注意力)
        channel_mask = self.channel_att(fused)

        # 空间注意力
        spatial_mask = self.spatial_att(fused)

        # 空间特征融合 - 输出out_channels通道
        spatial_mask_expanded = spatial_mask.repeat(1, self.in_channels, 1, 1)
        spatially_fusion = self.fusion(spatial_mask_expanded)

        # 通道加权特征 - 使用通道注意力结果
        channel_weighted = fused * channel_mask

        # 将通道加权特征降维
        channel_weighted_reduced = self.channel_reduce(channel_weighted)

        # 最终融合
        result = channel_weighted_reduced * spatially_fusion
        return torch.sigmoid(result)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        # 深度和RGB通道数
        self.inc_depth2 = inc_depth2
        self.inc_rgb = inc_rgb

        # 自动计算融合模块输入通道 (深度+RGB)
        fusion_in_channels = inc_depth2 + inc_rgb

        # 关键修复：为CCFF提供out_channels参数
        self.fus_atten = CCFF(in_channels=fusion_in_channels, out_channels=inc_depth2)

        self.conv1 = nn.Conv2d(inc_depth2, inc_depth2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(inc_depth2)

        self.cross_atten = Cross_Atten_Lite_split(inc_depth2, inc_rgb)

    def forward(self, depth_out, rgb_out):
        # 特征拼接
        fused_features = torch.cat([depth_out, rgb_out], dim=1)

        # 特征融合注意力
        fused_attention = self.fus_atten(fused_features)

        # 调整通道
        fused_attention = self.conv1(fused_attention)
        fused_attention = self.bn(fused_attention)

        # 跨模态注意力
        fused_attention = self.cross_atten(fused_attention, depth_out, rgb_out)

        return fused_attention


class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num):
        super(down_sample_block, self).__init__()
        self.block_num = block_num

        if block_num != 0:
            self.depth_stem = stem2[block_num]
            self.rgb_stem = stem1[block_num]
        else:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = stem1[0]

        self.rgb_layer = layers1[block_num]
        self.depth_layer = layers2[block_num]

        self.depth_norm = norm2[block_num]

        if self.block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb)

    def forward(self, image, depth):
        B = image.shape[0]
        image = self.rgb_stem(image)
        rgb_out = self.rgb_layer(image)

        depth_out, H, W = self.depth_stem(depth)

        for i, blk in enumerate(self.depth_layer):
            depth_out = blk(depth_out, H, W)
        depth_out = self.depth_norm(depth_out)
        depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # SCC_Ablation
        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        else:
            return rgb_out, depth_out


class B0_T(nn.Module):
    def __init__(self, num_classes=1):
        super(B0_T, self).__init__()

        self.channel = [32, 64, 160, 256]
        channel_list2 = [96, 192, 384, 768]

        self.down_sample_1 = down_sample_block(inc_depth=self.channel[0], inc_rgb=channel_list2[0], block_num=0)

        self.down_sample_2 = down_sample_block(inc_depth=self.channel[1],
                                               inc_rgb=channel_list2[1], block_num=1)

        self.down_sample_3 = down_sample_block(inc_depth=self.channel[2],
                                               inc_rgb=channel_list2[2], block_num=2)

        self.down_sample_4 = down_sample_block(inc_depth=self.channel[3],
                                               inc_rgb=channel_list2[3], block_num=3)

        self.Decoder = DecoderHead(in_channels=self.channel, num_classes=num_classes, dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d,
                                   embed_dim=256)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)

        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        _, depth_out = self.down_sample_4(rgb_out, depth_out3)

        rgb_out = self.Decoder(
            [depth_out1,
             depth_out2,
             depth_out3,
             depth_out])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        return rgb_out


if __name__ == '__main__':
    model = B0_T(num_classes=1)
    model.eval()
    print(model)
    image = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)
    macs, params = profile(model, inputs=(image, depth,))
    print(macs / (1000 ** 3))
    print(params / (1000 ** 2))
