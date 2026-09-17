from torch.nn import functional as F
from src.mix_transformer import OverlapPatchEmbed, mit_b0
from src.convnext import convnext_tiny
from thop import profile
from src.MLPDecoder import DecoderHead
import os
import torch
from torch import nn


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


class MaskAttention(nn.Module):
    def __init__(self, channels, size):
        super(MaskAttention, self).__init__()
        self.channels = channels  # 输入的通道数
        self.size = size  # 输入图像的大小（height, width）

        # 定义查询（Q）、键（K）和值（V）的线性变换层
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.mask = None  # 初始化mask，暂时为空
        self.norm = nn.LayerNorm([channels])  # 对输出进行LayerNorm标准化
        self._init_weight()

    def forward(self, x):
        # 获取输入张量的批次大小、通道数、高度和宽度
        batch_size, channels, height, width = x.size()

        # 检查输入的通道数是否与初始化时的通道数一致
        if channels != self.channels:
            raise ValueError("Input channel size does not match initialized channel size.")

        # 将输入张量展平并转置为 (batch_size, height * width, channels) 形状
        x_flat = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # 计算查询（Q）、键（K）和值（V）
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)

        # 计算注意力分数（scaled dot-product）
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5)  # 对分数进行缩放

        # 如果mask为空或其尺寸与输入图像大小不匹配，则生成新的随机二值mask
        if self.mask is None or self.mask.size(-1) != height * width:
            # 创建一个二值mask，随机生成0或1的值
            binary_mask = torch.randint(0, 2, (batch_size, height, width), device=x.device)
            binary_mask = binary_mask.view(batch_size, -1)

            # 将大于0.5的值设为0，小于0.5的值设为负无穷（即遮蔽这些区域）
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device),
                                         torch.tensor(-float('inf'), device=x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width,
                                                           -1)  # 扩展为 (batch_size, height * width, height * width) 形状

        # 将mask添加到注意力分数中
        scores = scores + self.mask

        # 对分数进行softmax归一化，得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 使用注意力权重对值（V）进行加权求和，得到注意力输出
        attention_output = torch.matmul(attention_weights, V)

        # 将输入与注意力输出相加，并进行标准化
        attention_output = attention_output + x_flat
        attention_output = self.norm(attention_output)

        # 恢复为原始的 (batch_size, channels, height, width) 形状
        output = attention_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)


class MultiScaleSEBlock(nn.Module):
    """多尺度通道注意力机制 (参考RT-DETR的通道混合器设计)"""

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


def get_activation(act):
    if act == 'silu':
        return nn.SiLU()
    elif act == 'relu':
        return nn.ReLU()
    else:
        return nn.Identity()


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
    """结构重参数化卷积块 (参考RT-DETR的CCFF模块)"""

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
        att_weights = channel_weights * spatial_weights

        # 5. 应用注意力并融合特征
        att_fused = fused * att_weights
        fused_out = self.fusion_blocks(att_fused)

        # 6. 输出适配
        return self.output_conv(fused_out)


# Concatenate operation
class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        self.inc_depth2 = inc_depth2
        self.inc_rgb = inc_rgb

        # 使用CCFF模块替代SpatialAttention_max
        self.ccff = CCFF(
            in_channels=inc_depth2 + inc_rgb,  # 输入拼接后的通道数
            out_channels=inc_depth2,  # 输出通道数保持与深度特征相同
            num_blocks=1  # 使用1个融合块
        )

        # 使用MaskAttention替换Cross_Atten_Lite_split
        # 注意：这里size参数需要根据实际输入特征图大小动态设置，这里先设置为None
        self.mask_attention = MaskAttention(channels=inc_depth2, size=None)

    def forward(self, depth_out, rgb_out):
        # 使用CCFF模块融合深度和RGB特征
        fus_s = self.ccff(depth_out, rgb_out)

        # 应用MaskAttention机制
        fus_s = self.mask_attention(fus_s)

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