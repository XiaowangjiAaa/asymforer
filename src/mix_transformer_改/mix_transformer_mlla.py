import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE(nn.Module):
    """修复的旋转位置编码模块"""

    def __init__(self, dim):
        super(RoPE, self).__init__()
        self.dim = dim

    def forward(self, x):
        B, H, W, C = x.shape
        device, dtype = x.device, x.dtype

        # 生成位置网格
        h_range = torch.arange(H, device=device, dtype=dtype)
        w_range = torch.arange(W, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(h_range, w_range, indexing='ij')

        # 为每个特征维度生成旋转频率
        dim_range = torch.arange(self.dim // 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (dim_range / (self.dim // 2 - 1)))

        # 位置编码 (H, W, C//2)
        pos_h = grid_h.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        pos_w = grid_w.unsqueeze(-1) * inv_freq.view(1, 1, -1)

        # 计算旋转分量
        cos_h = torch.cos(pos_h)
        sin_h = torch.sin(pos_h)
        cos_w = torch.cos(pos_w)
        sin_w = torch.sin(pos_w)

        # 将输入分成实部和虚部
        x = x.view(B, H, W, 2, C // 2)
        x_re, x_im = x[:, :, :, 0], x[:, :, :, 1]

        # 应用旋转
        x_re_rot = x_re * cos_h - x_im * sin_h
        x_im_rot = x_re * sin_h + x_im * cos_h

        # 应用位置相关的加权
        x_re_rot = x_re_rot * cos_w - x_im_rot * sin_w
        x_im_rot = x_re_rot * sin_w + x_im_rot * cos_w

        # 合并结果
        x_rot = torch.stack([x_re_rot, x_im_rot], dim=3).flatten(3, 4)
        return x_rot


class MLLAttention(nn.Module):
    """带详细日志的混合线性注意力"""

    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
       

        assert dim % num_heads == 0, f"dim {dim} 应当能被 num_head {num_heads} 整除"
        assert self.head_dim % 2 == 0, f"头维度 {self.head_dim} 必须是偶数"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.rope = RoPE(dim=self.head_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape


        # 投影QKV
        qkv = self.qkv(x)


        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)


        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        # 应用ELU激活
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_combined = q.reshape(B * self.num_heads, N, self.head_dim)
        k_combined = k.reshape(B * self.num_heads, N, self.head_dim)

        # 重塑为空间格式 (B*num_heads, H, W, head_dim)
        q_space = q_combined.reshape(B * self.num_heads, H, W, self.head_dim)
        k_space = k_combined.reshape(B * self.num_heads, H, W, self.head_dim)
        # ==== 修复结束 ====

        # 应用RoPE位置编码
        q_rot = self.rope(q_space)
        k_rot = self.rope(k_space)

        # 恢复多头格式 (先还原序列维度，再分离多头)
        q_rot = q_rot.reshape(B * self.num_heads, H * W, self.head_dim)
        k_rot = k_rot.reshape(B * self.num_heads, H * W, self.head_dim)

        # 恢复原始维度 (B, num_heads, N, head_dim)
        q_rot = q_rot.reshape(B, self.num_heads, N, self.head_dim)
        k_rot = k_rot.reshape(B, self.num_heads, N, self.head_dim)
        # 准备空间格式用于RoPE


        # 线性注意力机制
        context = torch.einsum('bhnc,bhnd->bhcd', k_rot, v)
        attn = torch.einsum('bhnc,bhcd->bhnd', q_rot, context) * (self.head_dim ** -0.5)


        # 局部位置增强 (LEPE)
        v_space = v.permute(0, 2, 1, 3).reshape(B, H, W, -1)
        v_combined = v_space.permute(0, 3, 1, 2)
        lepe = self.lepe(v_combined)

        # 正确重塑LEPE输出
        lepe = lepe.permute(0, 2, 3, 1).reshape(B, N, self.num_heads, self.head_dim)
        lepe = lepe.permute(0, 2, 1, 3)


        attn = attn + lepe


        # 重组输出
        attn = attn.permute(0, 2, 1, 3).contiguous()


        attn = attn.reshape(B, N, -1)

        return attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MLLAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # 添加更详细的维度检查
        expected_numel = H * W
        actual_numel = x.shape[1]
        if expected_numel != actual_numel:
            raise ValueError(
                f"Block维度不匹配: 输入序列长度={actual_numel}, "
                f"但H×W={H}x{W}={expected_numel}. "
                "请检查前面的embedding层。"
            )
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                         embed_dims=[64, 128, 256, 512],
                         num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                         attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                         depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.img_size = to_2tuple(img_size)  # 存储为元组

                # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                      embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                      in_chans=embed_dims[0],
                                                      embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                      in_chans=embed_dims[1],
                                                      embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                      in_chans=embed_dims[2],
                                                      embed_dim=embed_dims[3])

                # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

                # 移除预设分辨率
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
            qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
            qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[cur + i])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
