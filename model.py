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

        self.mconv_1 = nn.Conv2d(4, 64, (3, 3), (1, 1), (1, 1))
        self.mconv_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.mconv_3 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.mbconv_1 = nn.Conv2d(64 * 3, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.mconv_4 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.mbconv_2 = nn.Conv2d(64 * 4, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.mconv_5 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.mbconv_3 = nn.Conv2d(64 * 5, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.mconv_6 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))

        self.mconv_7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.mconv_8 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.mconv_9 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.mbconv_4 = nn.Conv2d(64 * 3, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.mconv_10 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.mbconv_5 = nn.Conv2d(64 * 4, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.mconv_11 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))
        self.mbconv_6 = nn.Conv2d(64 * 5, 64 * 2, (1, 1), (1, 1), (0, 0))
        self.mconv_12 = nn.Conv2d(64 * 2, 64, (3, 3), (1, 1), (1, 1))

        self.deconv_1=nn.ConvTranspose2d(64*2,64,(3,3),(2,2),(1,1),(1,1))
        self.deconv_2 = nn.ConvTranspose2d(64, 64, (3, 3), (2, 2), (1, 1),(1,1))
        self.reconv=nn.Conv2d(64,4,(3,3),(1,1),(1,1))

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
        ms_2 = F.leaky_relu(self.mconv_2(ms_1))
        ms_3=F.leaky_relu(self.mconv_3(torch.cat((ms_1, ms_2), 1)))
        bms_1=F.leaky_relu(self.mbconv_1(torch.cat((ms_1, ms_2, ms_3), 1)))
        ms_4 = F.leaky_relu(self.mconv_4(bms_1))
        bms_2 = F.leaky_relu(self.mbconv_2(torch.cat((ms_1, ms_2, ms_3,ms_4), 1)))
        ms_5 = F.leaky_relu(self.mconv_5(bms_2))
        bms_3 = F.leaky_relu(self.mbconv_3(torch.cat((ms_1, ms_2, ms_3, ms_4,ms_5), 1)))
        ms_6 = F.leaky_relu(self.mconv_6(bms_3))

        ms_7 = F.leaky_relu(self.mconv_7(ms_6))
        ms_8 = F.leaky_relu(self.mconv_8(ms_7))
        ms_9=F.leaky_relu(self.mconv_9(torch.cat((ms_7, ms_8), 1)))
        bms_4=F.leaky_relu(self.mbconv_4(torch.cat((ms_7, ms_8, ms_9), 1)))
        ms_10 = F.leaky_relu(self.mconv_10(bms_4))
        bms_5 = F.leaky_relu(self.mbconv_5(torch.cat((ms_7, ms_8, ms_9,ms_10), 1)))
        ms_11 = F.leaky_relu(self.mconv_11(bms_5))
        bms_6 = F.leaky_relu(self.mbconv_6(torch.cat((ms_7, ms_8, ms_9, ms_10,ms_11), 1)))
        ms_12 = F.leaky_relu(self.mconv_12(bms_6))

        ms_up=F.leaky_relu(self.deconv_1(torch.cat((ms_6,ms_12),1)))
        ms_up=F.leaky_relu(self.deconv_2(ms_up))
        ms_up=F.leaky_relu(self.reconv(ms_up))
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