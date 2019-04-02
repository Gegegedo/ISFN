import torch.nn as nn
import torch.nn.functional as F
import torch
from ModelConfig import SR_Config

class dense_layer(nn.Sequential):
    def __init__(self,layer_index):
        super().__init__()
        self.add_module('conv',nn.Conv2d(SR_Config.input_channel+(layer_index-1)*SR_Config.growth_rate,SR_Config.growth_rate,3,1,1))
        self.add_module('relu',nn.ReLU())
    def forward(self, input):
        return torch.cat([input,super().forward(input)],1)
class dense_block(nn.Sequential):
    def __init__(self):
        super().__init__()
        for i in range(SR_Config.dense_layer_num):
            self.add_module('dense_layer%d'%(i+1),dense_layer(i+1))
class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.preconv=nn.Conv2d(4,SR_Config.input_channel,3,1,1)
        for i in range(SR_Config.block_num):
            self.add_module('dense_block%d'%(i+1),dense_block())
        self.bottle_neck=nn.Conv2d(SR_Config.dense_features+SR_Config.input_channel,SR_Config.bottle_neck_channel,1,1,0)
        self.deconv1=nn.ConvTranspose2d(SR_Config.bottle_neck_channel,SR_Config.bottle_neck_channel,3,2,1,1)
        self.deconv2=nn.ConvTranspose2d(SR_Config.bottle_neck_channel,SR_Config.bottle_neck_channel,3,2,1,1)
        self.rl=nn.Conv2d(SR_Config.bottle_neck_channel,4,3,1,1)
    def forward(self, ms):
        features=[]
        dense_layer_index=0
        features.append(F.relu(self.preconv(ms)))
        for modeule in self.children():
            if isinstance(modeule,dense_block):
                dense_layer_index+=1
                features.append(modeule(features[dense_layer_index-1]))
        neck_out=F.relu(self.bottle_neck(torch.cat(features,1)))
        deconv_out=F.relu(self.deconv1(neck_out))
        deconv_out=F.relu(self.deconv2(deconv_out))
        sr_out=F.relu(deconv_out)
        return sr_out
        # dense_feature=self.dense_block1(low_feature)
        # for dense_index in range(1,SR_Config.block_num):
        #     dense_feature=torch.cat(self.dense_block1(low_feature))
if __name__ == '__main__':
    net=SRNet()
    y=net.forward(torch.randn(64,4,64,64))