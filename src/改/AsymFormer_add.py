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

        self.fus_atten = SpatialAttention_max(in_channels=channel)
        self.conv1 = nn.Conv2d(channel, inc_depth2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(inc_depth2)

        self.cross_atten = Cross_Atten_Lite_split(inc_depth2, inc_rgb)

    def forward(self, depth_out, rgb_out):
        fus_s = torch.cat([depth_out, rgb_out], dim=1)
        fus_s = self.fus_atten(fus_s)
        fus_s = self.conv1(fus_s)
        fus_s = self.bn(fus_s)

        fus_s = self.cross_atten(fus_s, depth_out, rgb_out)

        return fus_s


class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num):
        super(down_sample_block, self).__init__()
        self.block_num = block_num

        if block_num == 0:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = stem1[0]
        elif block_num < 4:  # 处理0-3块
            self.depth_stem = stem2[block_num]
            self.rgb_stem = stem1[block_num]
        else:  # 新增第5块
            # 自定义第5块的下采样层
            self.depth_stem = OverlapPatchEmbed(
                in_chans=inc_depth,
                embed_dim=inc_depth * 2,  # 通道数翻倍
                patch_size=3,
                stride=2
            )
            self.rgb_stem = nn.Sequential(
                nn.Conv2d(inc_rgb, inc_rgb * 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(inc_rgb * 2),
                nn.GELU()
            )

        # 处理层定义
        if block_num < 4:
            self.rgb_layer = layers1[block_num]
            self.depth_layer = layers2[block_num]
            self.depth_norm = norm2[block_num]
        else:  # 第5块使用自定义层
            # 深度分支使用简化Transformer块
            self.depth_layer = nn.Sequential(
                nn.Linear(inc_depth * 2, inc_depth * 2),
                nn.GELU(),
                nn.Linear(inc_depth * 2, inc_depth * 2),
            )
            self.depth_norm = nn.LayerNorm(inc_depth * 2)

            # RGB分支使用简化卷积块
            self.rgb_layer = nn.Sequential(
                nn.Conv2d(inc_rgb * 2, inc_rgb * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(inc_rgb * 2),
                nn.GELU(),
                nn.Conv2d(inc_rgb * 2, inc_rgb * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(inc_rgb * 2),
                nn.GELU(),
            )

        if self.block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth * 2 if block_num == 4 else inc_depth,
                                  inc_rgb=inc_rgb * 2 if block_num == 4 else inc_rgb)

    def forward(self, image, depth):
        B = image.shape[0]

        # RGB分支处理
        image = self.rgb_stem(image)
        rgb_out = self.rgb_layer(image)

        # 深度分支处理
        if self.block_num < 4:
            depth_out, H, W = self.depth_stem(depth)
            for i, blk in enumerate(self.depth_layer):
                depth_out = blk(depth_out, H, W)
            depth_out = self.depth_norm(depth_out)
            depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        else:
            # 第5块的特殊处理
            depth_out, H, W = self.depth_stem(depth)
            depth_out = self.depth_layer(depth_out)
            depth_out = self.depth_norm(depth_out)
            depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        else:
            return rgb_out, depth_out


class B0_T(nn.Module):
    def __init__(self, num_classes=1):
        super(B0_T, self).__init__()

        # 扩展通道配置，增加第5层
        self.channel = [32, 64, 160, 256, 512]  # 新增512通道
        channel_list2 = [96, 192, 384, 768, 1536]  # 新增1536通道

        self.down_sample_1 = down_sample_block(inc_depth=self.channel[0], inc_rgb=channel_list2[0], block_num=0)
        self.down_sample_2 = down_sample_block(inc_depth=self.channel[1], inc_rgb=channel_list2[1], block_num=1)
        self.down_sample_3 = down_sample_block(inc_depth=self.channel[2], inc_rgb=channel_list2[2], block_num=2)
        self.down_sample_4 = down_sample_block(inc_depth=self.channel[3], inc_rgb=channel_list2[3], block_num=3)
        self.down_sample_5 = down_sample_block(inc_depth=self.channel[4], inc_rgb=channel_list2[4], block_num=4)  # 新增

        # 更新Decoder的输入通道数
        self.Decoder = DecoderHead(
            in_channels=self.channel,
            num_classes=num_classes,
            dropout_ratio=0.1,
            norm_layer=nn.BatchNorm2d,
            embed_dim=512  # 更新为最大通道数
        )

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)
        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        rgb_out, depth_out4 = self.down_sample_4(rgb_out, depth_out3)
        _, depth_out5 = self.down_sample_5(rgb_out, depth_out4)  # 新增

        # 将所有特征图传递给Decoder
        out = self.Decoder([
            depth_out1,
            depth_out2,
            depth_out3,
            depth_out4,
            depth_out5  # 新增
        ])
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    model = B0_T(num_classes=1)
    model.eval()
    print(model)
    image = torch.rand(1, 3, 480, 640)
    depth = torch.rand(1, 1, 480, 640)
    macs, params = profile(model, inputs=(image, depth,))
    print("Computational complexity (GMac):", macs / (1000 ** 3))
    print("Number of parameters (M):", params / (1000 ** 2))