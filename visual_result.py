import gdal
import numpy as np
import cv2
def chaneltransform(fusionimage):
    gdal.AllRegister()
    driver = gdal.GetDriverByName("GTiff")
    fusionimage = gdal.Open(fusionimage.encode('utf-8').decode(), gdal.GA_ReadOnly)
    im_width = fusionimage.RasterXSize
    im_height = fusionimage.RasterYSize
    # transformimage = os.path.join(uploadfiles[0], "chaneltransform.tif")
    # dstDS = driver.Create(transformimage,
    #                       xsize=im_width, ysize=im_height, bands=3, eType=gdal.GDT_Byte)
    visual_image = np.zeros(shape=(im_height, im_width, 3),dtype='uint8')
    for iband in range(1, 4):
        imgMatrix = fusionimage.GetRasterBand(iband).ReadAsArray(0, 0, im_width, im_height)
        zeros = np.size(imgMatrix) - np.count_nonzero(imgMatrix)
        minVal = np.percentile(imgMatrix, float(zeros / np.size(imgMatrix) * 100 + 0.15))
        maxVal = np.percentile(imgMatrix, 99)

        idx1 = imgMatrix < minVal
        idx2 = imgMatrix > maxVal
        idx3 = ~idx1 & ~idx2
        imgMatrix[idx1] = imgMatrix[idx1] * 20 / minVal
        imgMatrix[idx2] = 255
        idx1 = None
        idx2 = None
        imgMatrix[idx3] = pow((imgMatrix[idx3] - minVal) / (maxVal - minVal), 0.9) * 255
        if iband == 1:
            # dstDS.GetRasterBand(3).WriteArray(imgMatrix)
            # dstDS.FlushCache()
            visual_image[:, :, 2] = imgMatrix
            imgMatrix = None
        elif iband == 2:
            # dstDS.GetRasterBand(2).WriteArray(imgMatrix)
            # dstDS.FlushCache()
            visual_image[:, :, 1] = imgMatrix
            imgMatrix = None
        else:
            # dstDS.GetRasterBand(1).WriteArray(imgMatrix)
            # dstDS.FlushCache()
            visual_image[:, :, 0] = imgMatrix
            imgMatrix = None
    fusionimage = None
    dstDS = None
    return visual_image
if __name__ == '__main__':
    # visual_image=chaneltransform("fusion_downtown.tif")
    visual_image = chaneltransform("fusion_downtown.tif")
    origin=chaneltransform("ms_downtown.tif")
    cv2.imshow("",np.hstack((visual_image,origin)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()