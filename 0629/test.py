# -*- coding: utf-8 -*-
import cv2
import numpy as np
image = cv2.imread('images/ocean_sunset.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
original = cv2.imread('images/ocean_day.jpg')
original = cv2.cvtColor(original,cv2.COLOR_BGR2LAB)


def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:,:,0])
    image_std_l = np.std(image[:,:,0])
    image_avg_a = np.mean(image[:,:,1])
    image_std_a = np.std(image[:,:,1])
    image_avg_b = np.mean(image[:,:,2])
    image_std_b = np.std(image[:,:,2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg,std)

image_avg,image_std = getavgstd(image)
original_avg,original_std = getavgstd(original)

height,width,channel = image.shape
for i in range(0,height):
    for j in range(0,width):
        for k in range(0,channel):
            t = image[i,j,k]
            t = (t-image_avg[k])*(original_std[k]/image_std[k]) + original_avg[k]
            # t = 0 if t<0 else t
            # t = 255 if t>255 else t
            if t < 0:
                t = 0
            elif t > 255:
                t = 255
            else:
                t = t
            image[i,j,k] = t
image = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imwrite('out.jpg',image)