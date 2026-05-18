import torch
import torch.nn as nn
from headfactory import HeadFactory


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_channel = int(out_channels // 4)

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels)
        )

        self.se = SEBlock(out_channels)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.residual_conv(x)

        out = torch.cat([self.branch1x1(x), self.branch5x5(x), self.branch3x3(x), self.branch_pool(x)], dim=1)

        out += identity
        out = self.se(out)
        out = self.final_bn(out)
        return self.relu(out)


@HeadFactory.register("head2")
class InceptionTransformerHead(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        self.init_h = 24
        self.init_w = 32

        self.inception_blocks = nn.Sequential(
            InceptionResidualBlock(1, 128),
            InceptionResidualBlock(128, 256),
            InceptionResidualBlock(256, 512),
            InceptionResidualBlock(512, 768),
        )

        self.feature_reduce = nn.Conv2d(768, 256, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

        self.ln = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.init_h, self.init_w)
        x = self.inception_blocks(x)
        x = self.feature_reduce(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w).transpose(1, 2)

        x = self.transformer(x)  # -> [Batch, 768, 256]

        x = x.mean(dim=1)

        x = self.ln(x)
        x = self.dropout(x)
        return self.fc_out(x)
