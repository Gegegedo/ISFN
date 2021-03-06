from torch.utils.data import Dataset
import torch
import numpy as np
import gdal
import cv2
from functools import reduce
class Resample():
    def __init__(self,factor=4):
        self.factor=factor
    def __call__(self,imageinfo):
        #args:ms,pan
        if imageinfo.option!='Test':
            input_ms=cv2.resize(src=imageinfo.ms,dsize=(int(imageinfo.ms.shape[1]/self.factor),int(imageinfo.ms.shape[0]/self.factor)),
                                interpolation=cv2.INTER_LINEAR)
            input_pan=cv2.resize(src=imageinfo.pan,dsize=(int(imageinfo.pan.shape[1]/(self.factor)),int(imageinfo.pan.shape[0]/(self.factor))),
                                 interpolation=cv2.INTER_LINEAR)
            # input_ms=cv2.resize(src=input_ms,dsize=(imageinfo.ms.shape[1],imageinfo.ms.shape[0]),interpolation=cv2.INTER_LINEAR)
            imageinfo.label = imageinfo.ms.astype(np.float32)
            imageinfo.ms=input_ms.astype(np.float32)
            imageinfo.pan=input_pan.astype(np.float32)
        else:
            # input_pan = cv2.resize(src=imageinfo.pan,dsize=(int(imageinfo.ms.shape[1]), int(imageinfo.ms.shape[0])),
            #                       interpolation=cv2.INTER_LINEAR)
            imageinfo.pan=imageinfo.pan.astype(np.float32)
            imageinfo.ms=imageinfo.ms.astype(np.float32)
        return imageinfo
class Registration():
    pass
class Normalize():

    def __call__(self,imageinfo):
        #args:ms,pan,label
        # max_pixel = max((imageinfo.ms.max(), imageinfo.pan.max()))
        # min_pixel = min((imageinfo.ms.min(), imageinfo.pan.min()))
        # imageinfo.max_pixel = max_pixel
        # imageinfo.min_pixel = min_pixel
        imageinfo.ms = (imageinfo.ms - imageinfo.min_pixel) / (imageinfo.max_pixel - imageinfo.min_pixel)
        imageinfo.pan = (imageinfo.pan - imageinfo.min_pixel) / (imageinfo.max_pixel - imageinfo.min_pixel)
        if imageinfo.option!='Test':
            imageinfo.label=(imageinfo.label-imageinfo.min_pixel)/(imageinfo.max_pixel - imageinfo.min_pixel)
                   # cv2.normalize(src=args[2],dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return imageinfo
class ToTensor():
    def __call__(self,imageinfo):
        imageinfo.ms = torch.from_numpy(imageinfo.ms.transpose(2, 0, 1))
        imageinfo.pan = torch.from_numpy(np.expand_dims(imageinfo.pan, axis=0))
        if imageinfo.option!='Test':
            imageinfo.label=torch.from_numpy(imageinfo.label.transpose(2,0,1))
        return imageinfo.to_dict()
class ImageInfo():
    def __init__(self,x_off,y_off,ms,pan,option):
        self.x_off=x_off
        self.y_off=y_off
        self.ms=ms
        self.pan=pan
        self.option=option
        self.max_pixel=1500
        self.min_pixel=0
        self.label=0
    def to_dict(self):
        return {'x_off':self.x_off,'y_off':self.y_off,'ms':self.ms,'pan':self.pan,\
                'max_pixel':self.max_pixel,'min_pixel':self.min_pixel,'label':self.label}
class GF2(Dataset):
    def __init__(self,ms_path,pan_path,patch_size,
                 transfrom=None,option='Train'):
        self.patch_size = patch_size
        self.option = option
        self.transform = transfrom
        self.ms=list(map(gdal.Open,ms_path))
        self.pan=list(map(gdal.Open,pan_path))
        self.x_limit=list(map(lambda ms,pan:min(ms.RasterXSize,int(pan.RasterXSize/4)),self.ms,self.pan))
        self.y_limit=list(map(lambda ms,pan:min(ms.RasterYSize,int(pan.RasterYSize/4)),self.ms,self.pan))
        self.x_patchs=list(map(lambda x_limit:x_limit // self.patch_size,self.x_limit))
        self.y_patchs=list(map(lambda y_limit:y_limit // self.patch_size,self.y_limit))
        self.global_index=list(map(lambda x_patch,y_patch:x_patch*y_patch,self.x_patchs,self.y_patchs))
        for i in range(1,len(self.global_index)):
            self.global_index[i]=sum(self.global_index[0:i+1])
        # sample_xlocs=np.random.randint(low=0,high=x_limit-patch_size,size=patch_num)
        # sample_ylocs=np.random.randint(low=0,high=y_limit-patch_size,size=patch_num)
        # self.samples=list(zip(sample_xlocs,sample_ylocs))
    def __len__(self):
        # if self.option!='Val':
            return self.global_index[-1]
        # else:
        #     return 2000
    def __getitem__(self, idx):
        # if self.option=='Val':
        #     idx=idx*2
        source=-1
        for i,global_index in self.global_index:
            if source==-1:
                if idx<global_index:
                    source=i
            else:
                break
        x_off=(idx-self.global_index[source-1])%self.x_patchs[source]*self.patch_size
        y_off=(idx-self.global_index[source-1])//self.x_patchs[source]*self.patch_size
        ms=self.ms[source].ReadAsArray(x_off,y_off,self.patch_size,self.patch_size)
        pan=self.pan[source].ReadAsArray(int(x_off*4), int(y_off*4), self.patch_size*4, self.patch_size*4)
        #H*W*C
        ms=ms.transpose(1,2,0)
        if self.transform:
            return self.transform(ImageInfo(x_off,y_off,ms,pan,self.option))