import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResFCN256(nn.Module):
    def __init__(self, resolution=256, channel=3):
        super(ResFCN256, self).__init__()
        # --- ENCODER (ResNet-18 style) ---
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 256->128
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)  # 128
        self.layer2 = self._make_layer(128, 2, stride=2) # 128->64
        self.layer3 = self._make_layer(256, 2, stride=2) # 64->32
        self.layer4 = self._make_layer(512, 2, stride=2) # 32->16

        # --- DECODER (Transpose Convolutions) ---
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), # 16->32
            nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # 32->64
            nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64->128
            nn.BatchNorm2d(64), nn.ReLU())
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),   # 128->256
            nn.BatchNorm2d(32), nn.ReLU())
        
        self.final = nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid() # Output 0-1 range

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encode
        x = F.relu(self.bn1(self.conv1(x))) # 128
        x = self.layer1(x) # 128
        x = self.layer2(x) # 64
        x = self.layer3(x) # 32
        x = self.layer4(x) # 16
        
        # Decode
        x = self.up4(x) # 32
        x = self.up3(x) # 64
        x = self.up2(x) # 128
        x = self.up1(x) # 256
        
        x = self.final(x)
        x = self.sigmoid(x)
        return x