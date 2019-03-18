import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pconv_1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.pbn_1=nn.BatchNorm2d(64)
        self.mconv_1 = nn.Conv2d(4, 64, (5, 5), (1, 1), (2, 2))
        self.mbn_1 = nn.BatchNorm2d(64)
        self.fconv_1 = nn.Conv2d(128, 64, (5, 5), (1, 1), (2, 2))
        self.fbn_1 = nn.BatchNorm2d(64)

        self.pconv_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.pbn_2 = nn.BatchNorm2d(64)
        self.mconv_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.mbn_2 = nn.BatchNorm2d(64)
        self.fconv_2 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.fbn_2 = nn.BatchNorm2d(64)

        self.pconv_3 = nn.Conv2d(64*2, 64, (3, 3), (1, 1), (1, 1))
        self.pbn_3 = nn.BatchNorm2d(64)
        self.mconv_3 = nn.Conv2d(64*2, 64, (3, 3), (1, 1), (1, 1))
        self.mbn_3 = nn.BatchNorm2d(64)
        self.fconv_3 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.fbn_3 = nn.BatchNorm2d(64)

        self.fconv_4 = nn.Conv2d(64*3, int(64*3/2), (1, 1), (1, 1), (0, 0))
        self.fbn_4 = nn.BatchNorm2d(int(64*3/2))
        self.fconv_5 = nn.Conv2d(int(64*3/2), 4*4*4, (3, 3), (1, 1), (1, 1))
        self.fbn_5 = nn.BatchNorm2d(4*4*4)

        self.ps = nn.PixelShuffle(4)

    def forward(self, ms, pan):
        pan_1 = F.leaky_relu(self.pbn_1(self.pconv_1(pan)))
        ms_1 = F.leaky_relu(self.mbn_1(self.mconv_1(ms)))
        fusion_1 = F.leaky_relu(self.fbn_1(self.fconv_1(torch.cat((pan_1, ms_1), 1))))

        pan_2 = F.leaky_relu(self.pbn_2(self.pconv_2(pan_1)))
        ms_2 = F.leaky_relu(self.mbn_2(self.mconv_2(ms_1)))
        fusion_2 = F.leaky_relu(self.fbn_2(self.fconv_2(torch.cat((pan_2, ms_2), 1))))

        pan_3 = F.leaky_relu(self.pbn_3(self.pconv_3(torch.cat((pan_1, pan_2), 1))))
        ms_3 = F.leaky_relu(self.mbn_3(self.mconv_3(torch.cat((ms_1, ms_2), 1))))
        fusion_3 = F.leaky_relu(self.fbn_3(self.fconv_3(torch.cat((pan_3, ms_3), 1))))

        fusion = F.leaky_relu(self.fbn_4(self.fconv_4(torch.cat((fusion_1, fusion_2, fusion_3), 1))))
        fusion = self.ps(self.fbn_5(self.fconv_5(fusion)))

        return fusion