import numpy as np
import matplotlib.pyplot as plt
import cv2

# -----------------ロウコンストラクト------------------------------
#画像の読み込みと表示
low_contrast = cv2.imread('low_contrast.jpg',0)
# low_contrast -=  50
plt.imshow(low_contrast, cmap='gray')
plt.show()

#画像のヒストグラムを表示
n, bin, patches  = plt.hist(low_contrast.ravel(),range(256),facecolor='blue', alpha=0.5)
plt.show()

#計算
cumulative_bins = n
for i in range(1, len(cumulative_bins)):
    cumulative_bins[i] = cumulative_bins[i] + cumulative_bins[i-1]
plt.plot(range(len(n)),cumulative_bins)
plt.show()

#書き出し
# cv2.imwrite('original.jpg',low_contrast)
# cv2.imwrite('shift.jpg',low_contrast)

#----------------ハイコンストラクト----------------------------------
#ハイコントラストの画像
high_contrast = cv2.imread('high_contrast.png',0)
plt.imshow(high_contrast, cmap='gray')
plt.show()

#画像のヒストグラムを表示
n, bin, patches  = plt.hist(high_contrast.ravel(),range(256),facecolor='blue', alpha=0.5)
plt.show()

#計算
cumulative_bins = n
for i in range(1, len(cumulative_bins)):
    cumulative_bins[i] = cumulative_bins[i] + cumulative_bins[i-1]
plt.plot(range(len(n)),cumulative_bins)
plt.show()

#-------------------ヒストグラムの均一化-----------------------------
#ヒストグラム均一化
HE_image = cv2.equalizeHist(low_contrast)
plt.show()
n, bin, patches  = plt.hist(HE＿image.ravel(),range(256),facecolor='blue', alpha=0.5)
plt.show()
cumulative_bins = n
for i in range(1, len(cumulative_bins)):
    cumulative_bins[i] = cumulative_bins[i] + cumulative_bins[i-1]
plt.plot(range(len(n)),cumulative_bins)
cv2.imwrite('HE_image.jpg',HE_image)
