import torch.nn as nn
import torch.nn.functional as F
import torch
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pconv_1 = nn.Conv2d(5, 64, (3, 3), (1, 1), (1, 1))
        self.pconv_2 = nn.Conv2d(64,64,(3,3),(1,1),(1,1))
        self.pconv_3 = nn.Conv2d(64*2, 64, (3, 3), (1, 1), (1, 1))
        self.bconv_1 = nn.Conv2d(64 * 3, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.pconv_4 = nn.Conv2d(64*2, 64, (3, 3), (1, 1), (1, 1))
        self.bconv_2 = nn.Conv2d(64 * 4, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.pconv_5 = nn.Conv2d(64*2, 64, (3, 3), (1, 1), (1, 1))
        self.bconv_3 = nn.Conv2d(64 * 5, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.pconv_6 = nn.Conv2d(64*2, 4, (3, 3), (1, 1), (1, 1))


        self.mconv_1 = nn.Conv2d(4, 64, (5, 5), (1, 1), (2,2))
        self.mconv_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.mconv_3 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.ps_1 = nn.PixelShuffle(4)

        self.fconv_1 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.fconv_1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))


    def forward(self, ms, pan):
        # pan_1 = F.leaky_relu(self.pconv_1(pan))
        # pan_2 = F.leaky_relu(self.pconv_2(pan_1))
        # pan_3 = F.leaky_relu(self.pconv_3(torch.cat((pan_1,pan_2),1)))
        # bpan_1 = F.leaky_relu(self.bconv_1(torch.cat((pan_1, pan_2,pan_3), 1)))
        # pan_4 = F.leaky_relu(self.pconv_4(bpan_1))
        # bpan_2 = F.leaky_relu(self.bconv_2(torch.cat((pan_1, pan_2, pan_3,pan_4), 1)))
        # pan_5 = F.leaky_relu(self.pconv_5(bpan_2))
        # bpan_3 = F.leaky_relu(self.bconv_3(torch.cat((pan_1, pan_2, pan_3, pan_4,pan_5), 1)))
        # pan_6=F.leaky_relu(self.pconv_6(bpan_3))

        ms_1 = F.leaky_relu(self.mconv_1(ms))
        # fusion_1 = self.ps_1(F.leaky_relu(self.fconv_1(torch.cat((pan_1, ms_1), 1))))
        ms_2 = F.leaky_relu(self.mconv_2(ms_1))
        # fusion_2 = self.ps_2(F.leaky_relu(self.fconv_2(torch.cat((pan_2, ms_2), 1))))
        ms_3 = self.mconv_3(torch.cat((ms_1, ms_2), 1))
        ms_up = self.ps_1(ms_3)
        # fusion_3 = self.ps_3(F.leaky_relu(self.fconv_3(torch.cat((pan_3, ms_3), 1))))
        pan_1 = F.leaky_relu(self.pconv_1(torch.cat((pan,ms_up),1)))
        pan_2 = F.leaky_relu(self.pconv_2(pan_1))
        pan_3 = F.leaky_relu(self.pconv_3(torch.cat((pan_1, pan_2), 1)))
        bpan_1 = F.leaky_relu(self.bconv_1(torch.cat((pan_1, pan_2, pan_3), 1)))
        pan_4 = F.leaky_relu(self.pconv_4(bpan_1))
        bpan_2 = F.leaky_relu(self.bconv_2(torch.cat((pan_1, pan_2, pan_3, pan_4), 1)))
        pan_5 = F.leaky_relu(self.pconv_5(bpan_2))
        bpan_3 = F.leaky_relu(self.bconv_3(torch.cat((pan_1, pan_2, pan_3, pan_4, pan_5), 1)))
        pan_6 = F.leaky_relu(self.pconv_6(bpan_3))
        # fusion = F.leaky_relu(self.fconv_4(torch.cat((fusion_1, fusion_2, fusion_3), 1)))
        # fusion = F.leaky_relu(self.fconv_1(torch.cat((pan_6,ms_3),1)))
        # fusion = self.fconv_2(fusion)

        return pan_6