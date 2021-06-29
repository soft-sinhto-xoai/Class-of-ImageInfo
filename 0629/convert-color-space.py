import cv2
import numpy as np
import matplotlib.pyplot as plt

# BGR to RGB
img = cv2.imread('th.jpg')  #BGR color space
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # BGR -> RGB
plt.imshow(img)
# plt.show()

# GRAY LIGHTNESS
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
# plt.show()

def myRGB_to_Grayscalde_lightness(img):
    gray = ((img.max(axis=2) + img.min(axis=2)) / 2.0).astype(int)
    return gray

gray = myRGB_to_Grayscalde_lightness(img)
plt.imshow(gray, cmap='gray')
# plt.show()

# GRAY AVERAGE
def myRGB_to_Grayscale_average(img):
    gray = img.mean(axis=2)
    return gray

gray = myRGB_to_Grayscale_average(img)
plt.imshow(gray, cmap='gray')
# plt.show()

#GRAY LUMINOSITY
def myRGB_to_Grayscale_luminosity(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    gray = 0.21 * r + 0.72 * g + 0.07 * b
    return gray

gray = myRGB_to_Grayscale_luminosity(img)
plt.imshow(gray,cmap='gray')
# plt.show()

