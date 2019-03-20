from dataset import Resample,Normalize,ImageInfo
import gdal
from model import Net
import torch
import numpy as np
import cv2
import time
start=time.clock()
ms_path='ms_downtown.tif'
pan_path='pan_downtown.tif'
source_ms=gdal.Open(ms_path)
ms_array=source_ms.ReadAsArray(0,0,source_ms.RasterXSize,source_ms.RasterYSize)
ms_array=ms_array.transpose(1,2,0)
source_pan=gdal.Open(pan_path)
pan_array=source_pan.ReadAsArray(0,0,source_pan.RasterXSize,source_pan.RasterYSize)

ms_array=cv2.resize(src=ms_array,dsize=(128,128),interpolation=cv2.INTER_LINEAR)
pan_array=cv2.resize(src=pan_array,dsize=(512,512),interpolation=cv2.INTER_LINEAR)
image_info=ImageInfo(x_off=0,y_off=0,ms=ms_array,pan=pan_array,option='Test')

net = Net()
net.to('cuda:0')
load_dict=torch.load('best_model')
net.load_state_dict(load_dict)
net.eval()

r=Resample()
n=Normalize()
image_info=r(image_info)
image_info=n(image_info)
image_info.ms = image_info.ms.transpose(2, 0, 1)
image_info.ms = np.expand_dims(image_info.ms,0)
image_info.pan = np.expand_dims(image_info.pan, axis=0)
image_info.pan = np.expand_dims(image_info.pan, axis=0)

input_ms=torch.from_numpy(image_info.ms).to('cuda:0')
input_pan=torch.from_numpy(image_info.pan).to('cuda:0')

with torch.no_grad():
    predict_result=net.forward(input_ms,input_pan).to('cpu')
    predict_result = np.squeeze(predict_result.numpy())
    max_pixel = image_info.max_pixel
    min_pixel = image_info.min_pixel

    # img = cv2.normalize(src=predict_result.transpose(1, 2, 0), dst=None, alpha=min_pixel, \
    #                     beta=max_pixel, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    img=predict_result*image_info.max_pixel
    driver=gdal.GetDriverByName('GTiff')
    dst=driver.Create("fusion_downtown.tif",512,512,4,gdal.GDT_UInt16)
    for band in range(4):
        dst.GetRasterBand(band+1).WriteArray(img[band])
    del dst
print(time.clock()-start)
    # fusion[y_off[idx]:y_off[idx]+dataset.patch_size*4,x_off[idx]:x_off[idx]+dataset.patch_size*4,:]=img
    # fusion_visualize = cv2.normalize(src=img[:, :, 0:3], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                                  dtype=cv2.CV_8U)
    # cv2.imshow('', fusion_visualize)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()