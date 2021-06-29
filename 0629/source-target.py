# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

# OTHER COLOR SPACE
source_path = 'images/ocean_sunset.jpg'
target_path = 'images/ocean_day.jpg'

source = cv2.imread(source_path)
target = cv2.imread(target_path)

# source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
# plt.imshow(source)
# plt.show()
# plt.imsave("source.jpg", source)

# target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
# plt.imshow(target)
# plt.show()
# plt.imsave("target.jpg",target)

# change the color space
# image = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
image = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
# target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')
target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

# calculate function
def image_stats(lab_image):
    # l,a,b = cv2.split(lab_image)
    avg = []
    std = []
    lMean = np.mean(image[:,:,0])
    lStd  = np.std(image[:,:,0])
    aMean = np.mean(image[:,:,1])
    aStd  = np.std(image[:,:,1])
    bMean = np.mean(image[:,:,2])
    bStd = np.std(image[:,:,2])

    avg.append(lMean)
    avg.append(aMean)
    std.append(bMean)
    avg.append(lStd)
    std.append(aStd)
    std.append(bStd)
    return (avg, std)

# calculate
image_avg,image_std = image_stats(image)
target_avg,target_std = image_stats(target_lab)

# Add values

# (l,a,b) = cv2.split(target_lab)
# # l -= lMeanTar
# # l /= lStdTar
# # l *= lStdSrc
# # l += lMeanSrc
# l = ((l - lMeanTar)/lStdTar)* lStdSrc * lMeanSrc
# a = ((a - aMeanTar)/aStdTar) * aStdSrc * aMeanSrc
# b = ((b - bMeanTar)/bStdTar) * bStdSrc * bMeanSrc
# """
#     [np.clipの使い方]
#     第一引数aに処理する配列ndarray、第二引数a_minに最小値、第三引数a_maxに最大値を指定する。
#     最小値と最大値のどちらか一方のみを指定したい場合はNoneを使う。省略はできない。
# """
# l = np.clip(l, 0, 255)
# a = np.clip(a, 0, 255)
# b = np.clip(b, 0, 255)

# transfer = cv2.merge([l,a,b]) 
"""
    上のcv2.mergeがうまくいってなかったみたい
"""
height,width,channel = image.shape
for i in range(0,height):
    for j in range(0,width):
        for k in range(0,channel):
            t = image[i,j,k]
            t = (t-image_avg[k])*(target_std[k]/image_std[k]) + target_avg[k]
            # t = 0 if t<0 else t
            # t = 255 if t>255 else t
            if t < 0:
                t = 0
            elif t > 255:
                t = 255
            else:
                t = t
            image[i,j,k] = t

# transfer = cv2.cvtColor(transfer.astype('uint8'), cv2.COLOR_LAB2BGR)
# transfer = cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB)
# plt.imshow(cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB))
image = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# plt.imshow(transfer)
plt.imshow(image)
plt.show()
# plt.imsave("transfer.jpg",transfer)
