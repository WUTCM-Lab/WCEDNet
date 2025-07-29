import torch
import torch.nn as nn
from einops import rearrange

class DiffFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        return x

class WeightedCascadedEncoder(torch.nn.Module):
    def __init__(self, channels, group_num):
        super(WeightedCascadedEncoder, self).__init__()
        self.group_num = group_num
        self.attn = nn.ModuleList(
            SpatialSpectralCrossAttention(channels=channels // group_num, group_num=group_num) for i in range(self.group_num))

        # 可学习的权重参数
        self.group_weights = nn.Parameter(torch.ones(group_num) / group_num)

    def forward(self, x):  # x (B,C,H,W)
        feats_in = x.chunk(self.group_num, dim=2)
        feats_out = []
        feat = feats_in[0]
        normalized_weights = torch.softmax(self.group_weights, dim=0)
        for i in range(self.group_num):
            if i > 0:  # add the previous output to the input
                feat = feat * normalized_weights[i-1] + feats_in[i]
            feat = self.attn[i](feat)
            feats_out.append(feat)

        x1 = torch.cat(feats_out, 1)
        x = rearrange(x1, 'b c d h w -> b 1 (c d) h w')
        return x

class SpatialSpectralCrossAttention(nn.Module):

    def __init__(self, channels, group_num, band_kernel_size=11, cube_kernel_size=3,):
        super(SpatialSpectralCrossAttention, self).__init__()
        self.softmax = nn.Softmax(-1)

        self.dwconv_wd = nn.Conv3d(1, 1, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2))
        self.dwconv_hd = nn.Conv3d(1, 1, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0))
        self.dwconv_hw = nn.Conv3d(1, 1, kernel_size=(band_kernel_size, 1, 1), padding=(band_kernel_size // 2, 0, 0))

        # Sigmoid to normalize attention weights
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.dwconv_wd(x)
        x2 = self.dwconv_hd(x)
        x3 = self.dwconv_hw(x)

        # Normalize x2, x3, x4 to [0, 1] as attention weights
        x1_attn = self.sigmoid(x1)
        x2_attn = self.sigmoid(x2)
        x3_attn = self.sigmoid(x3)

        # Sequentially multiply x1 with x2, x3, x4 attention weights
        x_x1 = x * x1_attn  # x1 influenced by x2
        x_x1_x2 = x_x1 * x2_attn  # x1 influenced by x2 and x3
        x_x1_x2_x3 = x_x1_x2 * x3_attn  # x1 influenced by x2, x3, x4

        # Residual connection (optional)
        x_out = x_x1_x2_x3 + x

        return x_out

NUM_CLASS = 3

class Decoder(nn.Module):

    def __init__(self, in_features=100, num_classes=NUM_CLASS):
        super().__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*222, out_channels=100, kernel_size=(3, 3)),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        self.conv1d_features = nn.Sequential(
            nn.Conv1d(in_features, 1, kernel_size=1),
            nn.ReLU(),
        )

        # 分类器
        self.classifier1 = nn.Sequential(
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )


    def forward(self, x):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c d h w -> b (c d) h w')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        attn_weights = self.conv1d_features(x)  # (B, 1, seq_len)

        return self.classifier1(attn_weights.squeeze(1))

class WCEDNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.DF = DiffFeature()
        self.EN = WeightedCascadedEncoder(224, 8)
        self.de = Decoder()

    def forward(self, x1, x2):
        x = self.DF(x1.unsqueeze(1), x2.unsqueeze(1))
        x = self.EN(x)
        y = self.de(x)

        return y

