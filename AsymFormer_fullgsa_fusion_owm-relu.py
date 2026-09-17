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

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior




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
        (sin, cos), mask = rel_pos
        assert h * w == mask.size(3)
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
        self.embed_dim = embed_dim
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
        # 确保输入通道数与embed_dim一致
        if C != self.embed_dim:
            raise ValueError(f"Input channels {C} must match embed_dim {self.embed_dim}")
        # 生成几何先验
        geo_prior = self.geo_prior((H, W), depth_map)

        # 生成位置编码
        pos_h = torch.arange(H, device=x.device).float()
        pos_w = torch.arange(W, device=x.device).float()
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w) #indexing='ij'
        pos_grid = torch.stack((grid_h, grid_w), dim=-1)  # [H, W, 2]
        pos_embed = self.pos_proj(pos_grid)  # [H, W, C]
        #pos_embed = pos_embed.reshape(1, H, W, C).repeat(B, 1, 1, 1)
        pos_embed = pos_embed.reshape(1, H, W, self.embed_dim)  # [1, H, W, embed_dim]
        pos_embed = pos_embed.repeat(B, 1, 1, 1)  # [B, H, W, embed_dim]
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


"""
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
"""

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
    """跨尺度上下文特征融合模块（修复通道不匹配问题）"""

    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()

        self.in_channels = in_channels  # 输入拼接后的通道数 (512)
        # 通道注意力输入改为256通道（与实际输入一致）
        self.channel_att = MultiScaleSEBlock(in_channels)  # 关键修复点

        # 空间注意力输入改为256通道
        self.spatial_att = SpatialFrequencyAttention(in_channels)  # 关键修复点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 融合块保持512输入（拼接后通道数）
        self.RepVggBlock= nn.Sequential(
            RepVggBlock(in_channels,in_channels)
        )

        # 输出适配层
        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)


    def forward(self, depth_feat, rgb_feat):
        # 通道注意力增强（输入256通道）
        #channel_refined = self.channel_att(depth_feat) * rgb_feat
        channel_refined = torch.cat([depth_feat, rgb_feat], dim=1)
        channel_refined = self.avg_pool(channel_refined)
        channel_mask = self.channel_att(channel_refined)
        channel_mask_telu = channel_mask.sigmoid()
        # 空间注意力融合（输入256通道）
        spatial_mask = self.spatial_att(channel_refined)
        spatial_mask_expanded = spatial_mask.repeat(1, self.in_channels, 1, 1)  # [B, 512, H, W]
        fuse = spatial_mask_expanded+channel_mask

        # 多级特征融合（拼接后512通道）
        #fused = torch.cat([channel_refined, spatially_refined], dim=1)
        spatially_fusion = self.RepVggBlock(fuse)
        fused = (channel_refined * channel_mask_telu * spatially_fusion).sigmoid()
        return self.output_conv(fused)  # 关键修复


class SCC_Module(nn.Module):
    def __init__(self, inc_depth, inc_rgb):
        super().__init__()
        # 通道统一处理
        self.depth_conv = nn.Conv2d(inc_depth, 256, 1)
        self.rgb_conv = nn.Conv2d(inc_rgb, 256, 1)

        # 自动计算融合模块输入通道
        fusion_in_channels = 256 + 256  # depth+rgb特征拼接
        self.ccff = CCFF(fusion_in_channels, 256)

        # 几何自注意力保留
        self.gsa = GSA_Module(embed_dim=256)
        # 添加自适应池化层统一尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((60, 80))  # 调整为统一尺寸
    def forward(self, depth_out, rgb_out):
        # 通道统一
        depth_feat = self.depth_conv(depth_out)
        rgb_feat = self.rgb_conv(rgb_out)
        depth_feat = self.adaptive_pool(depth_feat)
        rgb_feat = self.adaptive_pool(rgb_feat)
        # CCFF跨尺度融合
        fused = self.ccff(depth_feat, rgb_feat)

        # 几何自注意力增强
        depth_map = depth_out.mean(dim=1, keepdim=True)
        fusion=self.gsa(fused, depth_map)
        return fusion



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
            self.SCC = SCC_Module(inc_depth=inc_depth, inc_rgb=inc_rgb)
            # 添加通道调整层
            if block_num > 0:  # 除第一阶段外
                self.channel_adjust = nn.Conv2d(256, inc_depth, kernel_size=1)
            else:
                self.channel_adjust = None
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
            merge = self.channel_adjust(merge) if self.channel_adjust else merge
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

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior




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
        (sin, cos), mask = rel_pos
        assert h * w == mask.size(3)
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
        self.embed_dim = embed_dim
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
        # 确保输入通道数与embed_dim一致
        if C != self.embed_dim:
            raise ValueError(f"Input channels {C} must match embed_dim {self.embed_dim}")
        # 生成几何先验
        geo_prior = self.geo_prior((H, W), depth_map)

        # 生成位置编码
        pos_h = torch.arange(H, device=x.device).float()
        pos_w = torch.arange(W, device=x.device).float()
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w) #indexing='ij'
        pos_grid = torch.stack((grid_h, grid_w), dim=-1)  # [H, W, 2]
        pos_embed = self.pos_proj(pos_grid)  # [H, W, C]
        #pos_embed = pos_embed.reshape(1, H, W, C).repeat(B, 1, 1, 1)
        pos_embed = pos_embed.reshape(1, H, W, self.embed_dim)  # [1, H, W, embed_dim]
        pos_embed = pos_embed.repeat(B, 1, 1, 1)  # [B, H, W, embed_dim]
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


"""
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
"""

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

class RepVggBlock(nn.Module):
    """结构重参数化卷积块 (参考RT-DETR的CCFF模块[1](@ref))"""

    def __init__(self, ch_in, ch_out, ):
        super().__init__()
        self.conv3x3 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv1x1 = ConvNormLayer(ch_in, ch_out, 1, 1, act=None)
        #self.act = nn.SiLU() #if act == 'silu' else nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)+self.conv1x1(x)
        return torch.sigmoid(x)


class CCFF(nn.Module):
    """跨尺度上下文特征融合模块（修复通道不匹配问题）"""

    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()

        self.in_channels = in_channels  # 输入拼接后的通道数 (512)
        # 通道注意力输入改为256通道（与实际输入一致）
        self.channel_att = MultiScaleSEBlock(in_channels)  # 关键修复点

        # 空间注意力输入改为256通道
        self.spatial_att = SpatialFrequencyAttention(in_channels)  # 关键修复点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 融合块保持512输入（拼接后通道数）
        self.RepVggBlock= nn.Sequential(
            RepVggBlock(in_channels,in_channels)
        )

        # 输出适配层
        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, depth_feat, rgb_feat):
        # 拼接特征 (256+256=512)
        fused = torch.cat([depth_feat, rgb_feat], dim=1)

        # 通道注意力应用于拼接后的特征
        channel_mask = self.channel_att(fused)
        channel_mask_telu = torch.sigmoid(channel_mask)

        # 空间注意力应用于拼接后的特征
        spatial_mask = self.spatial_att(fused)

        # 空间注意力需要复制为拼接后的通道数
        spatial_mask_expanded = spatial_mask.repeat(1, self.in_channels, 1, 1)

        # 空间特征融合
        spatially_fusion = self.RepVggBlock(spatial_mask_expanded)

        # 三部分特征融合
        result = fused * channel_mask_telu * spatially_fusion
        result = torch.relu(result)

        return self.output_conv(result)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth, inc_rgb):
        super().__init__()
        # 通道统一处理
        self.depth_conv = nn.Conv2d(inc_depth, 256, 1)
        self.rgb_conv = nn.Conv2d(inc_rgb, 256, 1)

        # 自动计算融合模块输入通道
        fusion_in_channels = 256 + 256  # depth+rgb特征拼接
        self.ccff = CCFF(fusion_in_channels, 256)

        # 几何自注意力保留
        self.gsa = GSA_Module(embed_dim=256)
        # 添加自适应池化层统一尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((60, 80))  # 调整为统一尺寸
    def forward(self, depth_out, rgb_out):
        # 通道统一
        depth_feat = self.depth_conv(depth_out)
        rgb_feat = self.rgb_conv(rgb_out)
        depth_feat = self.adaptive_pool(depth_feat)
        rgb_feat = self.adaptive_pool(rgb_feat)
        # CCFF跨尺度融合
        fused = self.ccff(depth_feat, rgb_feat)

        # 几何自注意力增强
        depth_map = depth_out.mean(dim=1, keepdim=True)
        fusion=self.gsa(fused, depth_map)
        return fusion



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
            self.SCC = SCC_Module(inc_depth=inc_depth, inc_rgb=inc_rgb)
            # 添加通道调整层
            if block_num > 0:  # 除第一阶段外
                self.channel_adjust = nn.Conv2d(256, inc_depth, kernel_size=1)
            else:
                self.channel_adjust = None
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
            merge = self.channel_adjust(merge) if self.channel_adjust else merge
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
    #print(model)
    image = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)
    macs, params = profile(model, inputs=(image, depth,))
    print(macs / (1000 ** 3))
    print(params / (1000 ** 2))
