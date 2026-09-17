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
class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


# 1.几何先验生成
class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads=8, initial_value=2, heads_range=4):
        super().__init__()
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

    def generate_1d_depth_decay(self, H, W, depth_grid):
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        return mask

    def generate_1d_decay(self, l):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_structural_contrast(self, depth_map, threshold=0.05):
        B, _, H, W = depth_map.shape
        d_flat = depth_map.view(B, -1, 1)
        diff = torch.abs(d_flat - d_flat.transpose(1, 2))
        contrast_mask = (diff < threshold).float()
        return contrast_mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)

        if split_or_not:
            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))

            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            contrast_mask = self.generate_structural_contrast(depth_map)  # [B, HW, HW]
            return (sin, cos), (mask_h, mask_w), contrast_mask
        else:
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d
            contrast_mask = self.generate_structural_contrast(depth_map)
            mask = mask.unsqueeze(0) + contrast_mask.unsqueeze(1)
            return (sin, cos), mask, None





# 2.几何先验注意力计算
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

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        """
        x: (b h w c)
        rel_pos: mask: (n l l)
        """
        bsz, h, w, _ = x.size()
        (sin, cos), mask, contrast_mask = rel_pos

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
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

class GSA_Module(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        # 几何先验生成器
        self.geo_prior = GeoPriorGen(embed_dim, num_heads)

        # 几何自注意力层
        self.gsa_attn = Full_GSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            value_factor=1
        )

        # 残差连接前的层归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 位置编码投影
        self.pos_proj = nn.Linear(2, embed_dim)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.constant_(self.pos_proj.bias, 0)

    def forward(self, x, depth_map):
        """
        x: 输入特征图 [B, C, H, W]
        depth_map: 深度图 [B, 1, H, W]
        """
        B, C, H, W = x.size()

        # 生成几何先验
        geo_prior = self.geo_prior((H, W), depth_map)

        # 生成位置编码
        pos_h = torch.arange(H, device=x.device).float()
        pos_w = torch.arange(W, device=x.device).float()
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w) #indexing='ij'
        pos_grid = torch.stack((grid_h, grid_w), dim=-1)  # [H, W, 2]
        pos_embed = self.pos_proj(pos_grid)  # [H, W, C]
        pos_embed = pos_embed.reshape(1, H, W, C).repeat(B, 1, 1, 1)

        # 特征图与位置编码融合
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x + pos_embed

        # 层归一化
        x = self.norm(x)

        # 几何自注意力
        attn_out = self.gsa_attn(x, geo_prior)

        # 残差连接
        output = x + attn_out
        return output.permute(0, 3, 1, 2)  # [B, C, H, W]



class SpatialAttention_max(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_max, self).__init__()
        self.inc = torch.tensor(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_spatial = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction1, in_channels, bias=False),
        )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )

        self._init_weight()

    def forward(self, x):

        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)

        y_spatial = self.fc_spatial(y_avg).view(b, c, 1, 1)
        y_channel = self.fc_channel(y_avg).view(b, c, 1, 1)
        y_channel = y_channel.sigmoid()

        map = (x * (y_spatial)).sum(dim=1) / self.inc
        map = (map / self.inc).sigmoid().unsqueeze(dim=1)
        return map * x * y_channel

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        channel = inc_rgb + inc_depth2

                # 特征融合注意力
        self.fus_atten = SpatialAttention_max(in_channels=channel)
        self.conv1 = nn.Conv2d(channel, inc_depth2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(inc_depth2)

                # 替换为GSA模块
        self.gsa = GSA_Module(embed_dim=inc_depth2)

    def forward(self, depth_out, rgb_out):
                # 特征融合
        fus_s = torch.cat([depth_out, rgb_out], dim=1)
        fus_s = self.fus_atten(fus_s)
        fus_s = self.conv1(fus_s)
        fus_s = self.bn(fus_s)

                # 使用深度图作为几何先验
        depth_map = depth_out.mean(dim=1, keepdim=True)  # 生成伪深度图

                # 应用几何自注意力
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

        # SCC_Ablation
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
        #print("几何先验加载成功！")
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
    model = B0_T(num_classes=40)
    model.eval()
    print(model)
    image = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)
    macs, params = profile(model, inputs=(image, depth,))
    print(macs / (1000 ** 3))
    print(params / (1000 ** 2))
