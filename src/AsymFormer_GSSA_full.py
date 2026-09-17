from torch.nn import functional as F
from src.mix_transformer import OverlapPatchEmbed, mit_b0
from src.convnext import convnext_tiny
from thop import profile
from src.MLPDecoder import DecoderHead
import os
import math
from typing import Tuple


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
layers1 = [ft1[0], ft1[1], ft1[2], ft1[3]]

model2 = mit_b0()
layers2 = [model2.block1, model2.block2, model2.block3, model2.block4]
stem2 = [model2.patch_embed1, model2.patch_embed2, model2.patch_embed3, model2.patch_embed4]
norm2 = [model2.norm1, model2.norm2, model2.norm3, model2.norm4]

import torch
from torch import nn


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

    def forward(self, x: torch.Tensor):
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


# =====================================================================
# 1. 几何先验生成 (Geometry Prior Construction, 3.3.1)
# =====================================================================
class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads=8, initial_value=2, heads_range=4, struct_threshold=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.struct_threshold = struct_threshold

        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()

        # weight[0] -> spatial prior, weight[1] -> depth prior
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)

        # decay coefficient beta in (0,1), kept in log space (eq.9)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    # --- full 2D priors ---
    def generate_depth_decay(self, H: int, W: int, depth_grid):
        # eq.(4): pairwise depth distance matrix D in R^(HW x HW)
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)                       # [B, HW, HW]
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]  # [B, heads, HW, HW]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        # eq.(6): spatial prior S (Manhattan distance) in R^(HW x HW)
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid(index_h, index_w)
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)                          # [HW, HW]
        mask = mask * self.decay[:, None, None]                  # [heads, HW, HW]
        return mask

    def generate_structural_contrast(self, depth_map):
        # eq.(5): structural contrast mask M_struct in {0,1}
        B, _, H, W = depth_map.shape
        d_flat = depth_map.reshape(B, -1, 1)
        diff = (d_flat - d_flat.transpose(1, 2)).abs()
        return (diff < self.struct_threshold).float()            # [B, HW, HW]

    # --- axial (1D) priors ---
    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = (index[:, None] - index[None, :]).abs()
        return mask * self.decay[:, None, None]                  # [heads, l, l]

    def generate_1d_depth_decay(self, depth_line):
        diff = depth_line[:, :, None] - depth_line[:, None, :]
        diff = diff.abs()                                        # [B, l, l]
        return diff.unsqueeze(1) * self.decay[None, :, None, None]  # [B, heads, l, l]

    def generate_1d_structural_contrast(self, depth_line):
        diff = (depth_line[:, :, None] - depth_line[:, None, :]).abs()
        return (diff < self.struct_threshold).float()            # [B, l, l]

    def forward(self, HW_tuple: Tuple[int], depth_map):
        H, W = HW_tuple
        depth_map = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False)

        # rotary angle for full 2D attention
        index = torch.arange(H * W).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(H, W, -1)

        # eq.(7): G = w_s * S + w_d * D, then enhanced by structural contrast
        mask_pos = self.generate_pos_decay(H, W)                 # [heads, HW, HW]
        mask_d = self.generate_depth_decay(H, W, depth_map)      # [B, heads, HW, HW]
        mask = self.weight[0] * mask_pos + self.weight[1] * mask_d  # [B, heads, HW, HW]
        contrast = self.generate_structural_contrast(depth_map)  # [B, HW, HW]
        mask = mask + contrast.unsqueeze(1)                      # [B, heads, HW, HW]

        # eq.(10)(11): axial decomposition into G_x (row/width) and G_y (column/height)
        sin_h = torch.sin(torch.arange(H).to(self.decay)[:, None] * self.angle[None, :])
        cos_h = torch.cos(torch.arange(H).to(self.decay)[:, None] * self.angle[None, :])
        sin_w = torch.sin(torch.arange(W).to(self.decay)[:, None] * self.angle[None, :])
        cos_w = torch.cos(torch.arange(W).to(self.decay)[:, None] * self.angle[None, :])

        depth_col = depth_map.squeeze(1).mean(dim=1)             # [B, W] (mean over H)
        depth_row = depth_map.squeeze(1).mean(dim=2)             # [B, H] (mean over W)

        mask_w = self.weight[0] * self.generate_1d_decay(W) + \
                 self.weight[1] * self.generate_1d_depth_decay(depth_col)  # [B, heads, W, W]
        mask_w = mask_w + self.generate_1d_structural_contrast(depth_col).unsqueeze(1)

        mask_h = self.weight[0] * self.generate_1d_decay(H) + \
                 self.weight[1] * self.generate_1d_depth_decay(depth_row)  # [B, heads, H, H]
        mask_h = mask_h + self.generate_1d_structural_contrast(depth_row).unsqueeze(1)

        prior = {
            'sin': sin, 'cos': cos,          # [H, W, d]
            'mask': mask,                    # [B, heads, HW, HW]
            'sin_h': sin_h, 'cos_h': cos_h,  # [H, d]
            'sin_w': sin_w, 'cos_w': cos_w,  # [W, d]
            'mask_h': mask_h,                # [B, heads, H, H]
            'mask_w': mask_w,                # [B, heads, W, W]
        }
        return prior


# =====================================================================
# 2. 几何感知自注意力 (Geometry-aware Attention, 3.3.2 + 3.3.3)
# =====================================================================
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

    def forward_full(self, x: torch.Tensor, prior):
        # full 2D attention over HW x HW
        bsz, h, w, _ = x.size()
        sin, cos = prior['sin'], prior['cos']
        mask = prior['mask']
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

    def forward_axial(self, x: torch.Tensor, prior):
        # sequential row (W) then column (H) attention, eq.(10)(11)
        B, H, W, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling

        q = q.view(B, H, W, self.num_heads, self.key_dim)
        k = k.view(B, H, W, self.num_heads, self.key_dim)
        v = v.view(B, H, W, self.num_heads, self.head_dim)

        # --- row attention over W ---
        qr = q.permute(0, 1, 3, 2, 4)                            # [B,H,heads,W,d]
        qr = angle_transform(qr, prior['sin_w'], prior['cos_w'])
        kr = k.permute(0, 1, 3, 2, 4)
        kr = angle_transform(kr, prior['sin_w'], prior['cos_w'])
        vr = v.permute(0, 1, 3, 2, 4)                            # [B,H,heads,W,head_dim]
        qk_w = qr @ kr.transpose(-1, -2) + prior['mask_w'][:, None, :, :, :]
        attn_w = torch.softmax(qk_w, -1)
        x_w = attn_w @ vr                                         # [B,H,heads,W,head_dim]
        x_w = x_w.permute(0, 1, 3, 2, 4).reshape(B, H, W, -1)

        # --- column attention over H ---
        qh = q.permute(0, 2, 3, 1, 4)                            # [B,W,heads,H,d]
        qh = angle_transform(qh, prior['sin_h'], prior['cos_h'])
        kh = k.permute(0, 2, 3, 1, 4)
        kh = angle_transform(kh, prior['sin_h'], prior['cos_h'])
        vh = v.permute(0, 2, 3, 1, 4)                            # [B,W,heads,H,head_dim]
        qk_h = qh @ kh.transpose(-1, -2) + prior['mask_h'][:, None, :, :, :]
        attn_h = torch.softmax(qk_h, -1)
        x_h = attn_h @ vh                                         # [B,W,heads,H,head_dim]
        x_h = x_h.permute(0, 3, 1, 2, 4).reshape(B, H, W, -1)

        output = x_w + x_h + lepe
        return self.out_proj(output)

    def forward(self, x: torch.Tensor, prior, use_axial=True):
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
    def __init__(self, embed_dim, num_heads=8, use_axial=True):
        super().__init__()
        self.use_axial = use_axial
        self.geo_prior = GeoPriorGen(embed_dim, num_heads)
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
        attn_out = self.gsa_attn(x, prior, use_axial=self.use_axial)
        output = x + attn_out
        return output.permute(0, 3, 1, 2)


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
        x_fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(x_fft)
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
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        self.in_channels = in_channels
        self.channel_att = MultiScaleSEBlock(in_channels)
        self.spatial_att = SpatialFrequencyAttention(in_channels)
        self.fusion_blocks = nn.Sequential(
            *[RepVggBlock(in_channels, in_channels) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, depth_feat, rgb_feat):
        fused = torch.cat([depth_feat, rgb_feat], dim=1)
        channel_weights = self.channel_att(fused)
        spatial_weights = self.spatial_att(fused)
        att_weights = fused * channel_weights * spatial_weights
        fused_out = self.fusion_blocks(att_weights)
        return self.output_conv(fused_out)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        self.inc_depth2 = inc_depth2
        self.inc_rgb = inc_rgb
        self.ccff = CCFF(
            in_channels=inc_depth2 + inc_rgb,
            out_channels=inc_depth2,
            num_blocks=1
        )
        self.gsa = GSA_Module(embed_dim=inc_depth2)

    def forward(self, depth_out, rgb_out):
        fus_s = self.ccff(depth_out, rgb_out)
        depth_map = depth_out.mean(dim=1, keepdim=True)
        fus_s = self.gsa(fus_s, depth_map)
        return fus_s


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

        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        else:
            return rgb_out, depth_out


class B0_T(nn.Module):
    def __init__(self, num_classes):
        super(B0_T, self).__init__()
        self.channel = [32, 64, 160, 256]
        channel_list2 = [96, 192, 384, 768]

        self.down_sample_1 = down_sample_block(inc_depth=self.channel[0], inc_rgb=channel_list2[0], block_num=0)
        self.down_sample_2 = down_sample_block(inc_depth=self.channel[1], inc_rgb=channel_list2[1], block_num=1)
        self.down_sample_3 = down_sample_block(inc_depth=self.channel[2], inc_rgb=channel_list2[2], block_num=2)
        self.down_sample_4 = down_sample_block(inc_depth=self.channel[3], inc_rgb=channel_list2[3], block_num=3)

        self.Decoder = DecoderHead(in_channels=self.channel, num_classes=num_classes, dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d, embed_dim=256)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)
        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        _, depth_out = self.down_sample_4(rgb_out, depth_out3)

        rgb_out = self.Decoder([depth_out1, depth_out2, depth_out3, depth_out])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        return rgb_out


if __name__ == '__main__':
    model = B0_T(num_classes=40)
    model.eval()
    print(model)
    image = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)
    macs, params = profile(model, inputs=(image, depth,))
    print(macs / (1000 ** 3))
    print(params / (1000 ** 2))
