import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class UNet_DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet_DoubleConv, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, t):
        return self.model(t)


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.down.append(UNet_DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up.append(UNet_DoubleConv(feature * 2, feature))

        self.bottleneck = UNet_DoubleConv(features[-1], features[-1] * 2)
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, t):

        skips_connections = []

        for down in self.down:
            t = down(t)
            skips_connections.append(t)
            t = self.pool(t)

        t = self.bottleneck(t)

        skips_connections = skips_connections[::-1]

        for i in range(0, len(self.up), 2):
            t = self.up[i](t)
            skips_connection = skips_connections[i//2]

            if t.shape != skips_connection.shape:
                t = TF.resize(t, skips_connection.shape[2:])

            c = torch.cat([skips_connection, t], dim=1)
            t = self.up[i+1](c)

        return self.output(t)
            
        