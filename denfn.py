import torch.nn as nn
import torch.nn.functional as F
from ModelConfig import Dense
import torch
class Dense_layer(nn.Sequential):
    def __init__(self,layer_index):
        super().__init__()
        self.add_module("conv",nn.Conv2d(Dense.growth_rate*layer_index,Dense.growth_rate,3,1,1))
        self.add_module("relu",nn.ReLU())
    def forward(self, input):
        return torch.cat([super().forward(),input],1)
class Dense_extraction(nn.Sequential):
    def __init__(self):
        super().__init__()
        for layer_index in range(Dense.layer_num):
            self.add_module("dense_conv%d"%(layer_index+1),Dense_layer(layer_index))
        self.add_module("bottle_neck",nn.Conv2d(Dense.dense_channel,Dense.neck_channel,1,1))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ###MS feature extraction
        self.add_module("ms_conv",nn.Conv2d(4,Dense.growth_rate,3,1,1))
        self.add_module("ms_dense",Dense_extraction())
        self.add_module("deconv1",nn.ConvTranspose2d(Dense.neck_channel,Dense.neck_channel,3,2,1,1))
        self.add_module("deconv1", nn.ConvTranspose2d(Dense.neck_channel, Dense.neck_channel,3, 2, 1, 1))
        ###PAN feature extraction
        self.add_module("pan_conv",nn.Conv2d(1,Dense.growth_rate,3,1,1))
        self.add_module("pan_dense",Dense_extraction())
        ###feature fusion
        self.add_module("f_conv1",nn.Conv2d(Dense.neck_channel*2,Dense.neck_channel*2,3,1,1))
        self.add_module("f_conv2", nn.Conv2d(Dense.neck_channel * 2, Dense.neck_channel * 2, 3, 1, 1))
        self.add_module("f_conv3", nn.Conv2d(Dense.neck_channel * 2, 4, 3, 1, 1))
    def forward(self, ms,pan):
        ###MS feature extraction
        dense_ms_input=F.relu(self.ms_conv(ms))
        dense_ms_features=F.relu(self.ms_dense(dense_ms_input))
        dense_ms_features=F.relu(self.deconv1(dense_ms_features))
        dense_ms_features = F.relu(self.deconv2(dense_ms_features))
        ###PAN feature extraction
        dense_pan_input = F.relu(self.pan_conv(pan))
        dense_pan_features = F.relu(self.pan_dense(dense_pan_input))
        fusion=F.relu(self.f_conv1(torch.cat([dense_ms_features,dense_pan_features],1)))
        fusion = F.relu(self.f_conv2(fusion))
        fusion = F.relu(self.f_conv3(fusion))
        return fusion