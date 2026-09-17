from dataclasses import dataclass
from typing import Tuple

from torch.nn import functional as F
from src.mix_transformer import OverlapPatchEmbed, mit_b0
from src.convnext import convnext_tiny
from thop import profile
from src.MLPDecoder import DecoderHead
import os
import math

import torch
from torch import nn


# ---------------------------------------------------------------------
# 消融配置：每一项对应一个可独立开关的网络组件
# ---------------------------------------------------------------------
@dataclass
class AblationConfig:
    use_gsa: bool = True             # GSSA 几何结构自注意力
    fusion: str = "ccff"             # 融合方式: "ccff" | "cat"
    rgb_only: bool = False           # 仅 RGB 单分支
    decoder: str = "mlp"             # 解码器: "mlp" | "simple"
    use_channel_att: bool = True     # CCFF 通道注意力
    use_spatial_att: bool = True     # CCFF 频域空间注意力
    use_repvgg: bool = True          # CCFF 重参数化卷积
    use_struct: bool = True          # GSSA 结构对比掩码
    use_depth_prior: bool = True     # GSSA 深度先验
    use_spatial_prior: bool = True   # GSSA 空间先验
    use_axial: bool = True           # GSSA 轴向分解 (否则全 2D)


# ---------------------------------------------------------------------
# 命名消融实验注册表
# ---------------------------------------------------------------------
ABLATIONS = {
    # Group 1: 核心模块 + 解码器
    "full": dict(),
    "wo_gsa": dict(use_gsa=False),
    "wo_ciff": dict(fusion="cat"),
    "rgb_only": dict(rgb_only=True),
    "simple_decoder": dict(decoder="simple"),
    # Group 2: CIFF 内部
    "wo_channel_att": dict(use_channel_att=False),
    "wo_spatial_att": dict(use_spatial_att=False),
    "wo_repvgg": dict(use_repvgg=False),
    # Group 3: GSSA 几何先验
    "wo_struct": dict(use_struct=False),
    "wo_depth_prior": dict(use_depth_prior=False),
    "wo_spatial_prior": dict(use_spatial_prior=False),
    # Group 4: GSSA 注意力机制
    "full_2d": dict(use_axial=False),
}


def build_model(name, num_classes):
    cfg = AblationConfig(**ABLATIONS[name])
    return B0_T(num_classes, cfg), cfg


model1 = convnext_tiny(pretrained=True, drop_path_rate=0.3)
ft1 = model1.stages
stem = model1.downsample_layers
stem1 = [stem[0], stem[1], stem[2], stem[3]]
layers1 = [ft1[0], ft1[1], ft1[2], ft1[3]]

model2 = mit_b0()
layers2 = [model2.block1, model2.block2, model2.block3, model2.block4]
stem2 = [model2.patch_embed1, model2.patch_embed2, model2.patch_embed3, model2.patch_embed4]
norm2 = [model2.norm1, model2.norm2, model2.norm3, model2.norm4]


def channel_shuffle(x, groups: int):
    batchsize, N, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, N, groups, channels_per_group)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(batchsize, N, -1)
    return x


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(x, sin, cos):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


def get_activation(act):
    if act is None:
        return nn.Identity()
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    if act == 'silu':
        return nn.SiLU()
    if act == 'gelu':
        return nn.GELU()
    raise ValueError(act)


# ---------------------------------------------------------------------
# GSSA: 几何先验生成 + 几何感知自注意力
# ---------------------------------------------------------------------
class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, cfg, num_heads=8, initial_value=2, heads_range=4, struct_threshold=0.05):
        super().__init__()
        self.cfg = cfg
        self.num_heads = num_heads
        self.struct_threshold = struct_threshold

        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H, W, depth_grid):
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H, W):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid(index_h, index_w)
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_structural_contrast(self, depth_map):
        B, _, H, W = depth_map.shape
        d_flat = depth_map.reshape(B, -1, 1)
        diff = (d_flat - d_flat.transpose(1, 2)).abs()
        return (diff < self.struct_threshold).float()

    def generate_1d_decay(self, l):
        index = torch.arange(l).to(self.decay)
        mask = (index[:, None] - index[None, :]).abs()
        return mask * self.decay[:, None, None]

    def generate_1d_depth_decay(self, depth_line):
        diff = depth_line[:, :, None] - depth_line[:, None, :]
        diff = diff.abs()
        return diff.unsqueeze(1) * self.decay[None, :, None, None]

    def generate_1d_structural_contrast(self, depth_line):
        diff = (depth_line[:, :, None] - depth_line[:, None, :]).abs()
        return (diff < self.struct_threshold).float()

    def _full_mask(self, H, W, depth_map):
        mask = None
        if self.cfg.use_spatial_prior:
            mask = self.weight[0] * self.generate_pos_decay(H, W)
        if self.cfg.use_depth_prior:
            m = self.weight[1] * self.generate_depth_decay(H, W, depth_map)
            mask = m if mask is None else mask + m
        if self.cfg.use_struct:
            m = self.generate_structural_contrast(depth_map).unsqueeze(1)
            mask = m if mask is None else mask + m
        return mask

    def _axial_mask(self, l, depth_line):
        mask = None
        if self.cfg.use_spatial_prior:
            mask = self.weight[0] * self.generate_1d_decay(l)
        if self.cfg.use_depth_prior:
            m = self.weight[1] * self.generate_1d_depth_decay(depth_line)
            mask = m if mask is None else mask + m
        if self.cfg.use_struct:
            m = self.generate_1d_structural_contrast(depth_line).unsqueeze(1)
            mask = m if mask is None else mask + m
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map):
        H, W = HW_tuple
        depth_map = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False)

        if self.cfg.use_axial:
            sin_h = torch.sin(torch.arange(H).to(self.decay)[:, None] * self.angle[None, :])
            cos_h = torch.cos(torch.arange(H).to(self.decay)[:, None] * self.angle[None, :])
            sin_w = torch.sin(torch.arange(W).to(self.decay)[:, None] * self.angle[None, :])
            cos_w = torch.cos(torch.arange(W).to(self.decay)[:, None] * self.angle[None, :])
            depth_col = depth_map.squeeze(1).mean(dim=1)
            depth_row = depth_map.squeeze(1).mean(dim=2)
            prior = {
                'sin_h': sin_h, 'cos_h': cos_h,
                'sin_w': sin_w, 'cos_w': cos_w,
                'mask_h': self._axial_mask(H, depth_row),
                'mask_w': self._axial_mask(W, depth_col),
            }
        else:
            index = torch.arange(H * W).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
            prior = {'sin': sin, 'cos': cos, 'mask': self._full_mask(H, W, depth_map)}
        return prior


class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward_full(self, x, prior):
        bsz, h, w, _ = x.size()
        sin, cos, mask = prior['sin'], prior['cos'], prior['mask']
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2) + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = qk_mat @ vr
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        return self.out_proj(output)

    def forward_axial(self, x, prior):
        B, H, W, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling

        q = q.view(B, H, W, self.num_heads, self.key_dim)
        k = k.view(B, H, W, self.num_heads, self.key_dim)
        v = v.view(B, H, W, self.num_heads, self.head_dim)

        # row attention over W
        qr = q.permute(0, 1, 3, 2, 4)
        qr = angle_transform(qr, prior['sin_w'], prior['cos_w'])
        kr = k.permute(0, 1, 3, 2, 4)
        kr = angle_transform(kr, prior['sin_w'], prior['cos_w'])
        vr = v.permute(0, 1, 3, 2, 4)
        qk_w = qr @ kr.transpose(-1, -2) + prior['mask_w'][:, None, :, :, :]
        attn_w = torch.softmax(qk_w, -1)
        x_w = attn_w @ vr
        x_w = x_w.permute(0, 1, 3, 2, 4).reshape(B, H, W, -1)

        # column attention over H
        qh = q.permute(0, 2, 3, 1, 4)
        qh = angle_transform(qh, prior['sin_h'], prior['cos_h'])
        kh = k.permute(0, 2, 3, 1, 4)
        kh = angle_transform(kh, prior['sin_h'], prior['cos_h'])
        vh = v.permute(0, 2, 3, 1, 4)
        qk_h = qh @ kh.transpose(-1, -2) + prior['mask_h'][:, None, :, :, :]
        attn_h = torch.softmax(qk_h, -1)
        x_h = attn_h @ vh
        x_h = x_h.permute(0, 3, 1, 2, 4).reshape(B, H, W, -1)

        output = x_w + x_h + lepe
        return self.out_proj(output)

    def forward(self, x, prior, use_axial=True):
        if use_axial:
            return self.forward_axial(x, prior)
        return self.forward_full(x, prior)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class GSA_Module(nn.Module):
    def __init__(self, embed_dim, cfg, num_heads=8):
        super().__init__()
        self.cfg = cfg
        self.geo_prior = GeoPriorGen(embed_dim, cfg, num_heads)
        self.gsa_attn = Full_GSA(embed_dim=embed_dim, num_heads=num_heads, value_factor=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_proj = nn.Linear(2, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.constant_(self.pos_proj.bias, 0)

    def forward(self, x, depth_map):
        B, C, H, W = x.size()
        prior = self.geo_prior((H, W), depth_map)
        pos_h = torch.arange(H, device=x.device).float()
        pos_w = torch.arange(W, device=x.device).float()
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        pos_grid = torch.stack((grid_h, grid_w), dim=-1)
        pos_embed = self.pos_proj(pos_grid).reshape(1, H, W, C).repeat(B, 1, 1, 1)

        x = x.permute(0, 2, 3, 1) + pos_embed
        x = self.norm(x)
        attn_out = self.gsa_attn(x, prior, use_axial=self.cfg.use_axial)
        output = x + attn_out
        return output.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------
# CIFF: 跨尺度上下文特征融合
# ---------------------------------------------------------------------
class MultiScaleSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
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
        return channel_weights * x


class SpatialFrequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(in_channels, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        magnitude = torch.abs(torch.fft.rfft2(x, norm='ortho'))
        spatial_att = self.conv_spatial(x)
        magnitude_avg = magnitude.mean(dim=1, keepdim=True)
        magnitude_avg = F.interpolate(magnitude_avg, size=(H, W), mode='bilinear', align_corners=False)
        fused_att = spatial_att + magnitude_avg
        return self.sigmoid(fused_att)


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='silu'):
        super().__init__()
        self.conv3x3 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv1x1 = ConvNormLayer(ch_in, ch_out, 1, 1, act=None)
        self.act = nn.SiLU() if act == 'silu' else nn.ReLU()

    def forward(self, x):
        return self.act(self.conv3x3(x) + self.conv1x1(x))


class CCFF(nn.Module):
    def __init__(self, in_channels, out_channels, cfg, num_blocks=3):
        super().__init__()
        self.cfg = cfg
        self.channel_att = MultiScaleSEBlock(in_channels)
        self.spatial_att = SpatialFrequencyAttention(in_channels)
        if self.cfg.use_repvgg:
            self.fusion_blocks = nn.Sequential(
                *[RepVggBlock(in_channels, in_channels) for _ in range(num_blocks)]
            )
        else:
            self.fusion_blocks = nn.Sequential(
                ConvNormLayer(in_channels, in_channels, 3, 1, padding=1)
            )
        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, depth_feat, rgb_feat):
        fused = torch.cat([depth_feat, rgb_feat], dim=1)
        att_weights = fused
        if self.cfg.use_channel_att:
            att_weights = att_weights * self.channel_att(fused)
        if self.cfg.use_spatial_att:
            att_weights = att_weights * self.spatial_att(fused)
        fused_out = self.fusion_blocks(att_weights)
        return self.output_conv(fused_out)


class CatFusion(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super().__init__()
        self.conv = nn.Conv2d(inc_depth2 + inc_rgb, inc_depth2, 1)

    def forward(self, depth_feat, rgb_feat):
        return self.conv(torch.cat([depth_feat, rgb_feat], dim=1))


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.fusion == "ccff":
            self.fusion = CCFF(inc_depth2 + inc_rgb, inc_depth2, cfg, num_blocks=1)
        else:
            self.fusion = CatFusion(inc_depth2, inc_rgb)
        if cfg.use_gsa:
            self.gsa = GSA_Module(embed_dim=inc_depth2, cfg=cfg)

    def forward(self, depth_out, rgb_out):
        fus_s = self.fusion(depth_out, rgb_out)
        if self.cfg.use_gsa:
            depth_map = depth_out.mean(dim=1, keepdim=True)
            fus_s = self.gsa(fus_s, depth_map)
        return fus_s


# ---------------------------------------------------------------------
# 解码器
# ---------------------------------------------------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels[-1], num_classes, 1)

    def forward(self, inputs):
        return self.conv(inputs[-1])


# ---------------------------------------------------------------------
# 主干
# ---------------------------------------------------------------------
class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num, cfg):
        super(down_sample_block, self).__init__()
        self.block_num = block_num
        self.cfg = cfg

        if block_num != 0:
            self.depth_stem = stem2[block_num]
            self.rgb_stem = stem1[block_num]
        else:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = stem1[0]

        self.rgb_layer = layers1[block_num]
        self.depth_layer = layers2[block_num]
        self.depth_norm = norm2[block_num]

        if cfg.rgb_only:
            self.rgb_proj = nn.Conv2d(inc_rgb, inc_depth, 1)
        elif block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb, cfg=cfg)

    def forward(self, image, depth):
        B = image.shape[0]
        image = self.rgb_stem(image)
        rgb_out = self.rgb_layer(image)

        if self.cfg.rgb_only:
            merge = self.rgb_proj(rgb_out)
            return rgb_out, merge

        depth_out, H, W = self.depth_stem(depth)
        for blk in self.depth_layer:
            depth_out = blk(depth_out, H, W)
        depth_out = self.depth_norm(depth_out)
        depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        return rgb_out, depth_out


class B0_T(nn.Module):
    def __init__(self, num_classes, cfg):
        super(B0_T, self).__init__()
        self.cfg = cfg
        self.channel = [32, 64, 160, 256]
        channel_list2 = [96, 192, 384, 768]

        self.down_sample_1 = down_sample_block(self.channel[0], channel_list2[0], 0, cfg)
        self.down_sample_2 = down_sample_block(self.channel[1], channel_list2[1], 1, cfg)
        self.down_sample_3 = down_sample_block(self.channel[2], channel_list2[2], 2, cfg)
        self.down_sample_4 = down_sample_block(self.channel[3], channel_list2[3], 3, cfg)

        if cfg.decoder == "mlp":
            self.Decoder = DecoderHead(in_channels=self.channel, num_classes=num_classes,
                                       dropout_ratio=0.1, norm_layer=nn.BatchNorm2d, embed_dim=256)
        else:
            self.Decoder = SimpleDecoder(in_channels=self.channel, num_classes=num_classes)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]
        rgb_out, d1 = self.down_sample_1(image, depth)
        rgb_out, d2 = self.down_sample_2(rgb_out, d1)
        rgb_out, d3 = self.down_sample_3(rgb_out, d2)
        _, d4 = self.down_sample_4(rgb_out, d3)
        out = self.Decoder([d1, d2, d3, d4])
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    for name in ABLATIONS:
        model, cfg = build_model(name, num_classes=40)
        model.eval()
        image = torch.rand(1, 3, 480, 640)
        depth = torch.rand(1, 1, 480, 640)
        macs, params = profile(model, inputs=(image, depth,))
        print("{:<18} FLOPs {:.3f}G  Params {:.2f}M  cfg={}".format(
            name, macs / 1e9, params / 1e6, cfg))
