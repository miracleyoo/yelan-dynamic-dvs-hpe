""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Down, Up, OutConv

class Unet(nn.Module):
    def __init__(self, in_ch=8, seq_len=16, bilinear=False, **kwargs):
        super(Unet, self).__init__()
        self.n_channels = in_ch
        self.out_ch = seq_len
        self.bilinear = bilinear

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.hid = 64
        self.outc = OutConv(self.hid, self.out_ch)

        self.score_conv = nn.Sequential(*[
            nn.Conv2d(self.hid, self.hid*4, kernel_size=1),
            nn.Conv2d(self.hid*4, self.hid*8, kernel_size=1)
            ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.hid*8 , self.out_ch)

        print(f"Output channel number: {self.out_ch}")

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        masks = self.outc(x)

        x = self.score_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        scores = torch.sigmoid(self.fc(x))
        return masks.squeeze(), scores

    def predict_mask(self, x):
        masks, scores = self.forward(x)
        masks = torch.sigmoid(masks)[:,0:1]
        return masks
    

class UNet_HPE(nn.Module):
    def __init__(self, in_ch=8, out_ch=13, **kwargs):
        super(UNet_HPE, self).__init__()

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        return out
    