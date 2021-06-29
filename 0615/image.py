# coding:utf-8
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2

#ファイルの入出力
file = open("file.txt","r")
for line in file:
    print(line)

file = open("file.txt", "w")
file.write("Hello world")
file.write("This is out new text file")
file.write("and this is another line.")
file.close()
print("終了")

#画像データの読み込み
img = cv2.imread("orange.jpg")
# print(img.shape)

#画像を表示する
_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(_img)
# plt.show()

#画像を保存する
plt.imsave("orange2.jpg", _img)

#画像をリサイズする
#print(img.shape)
#-->(y,x,channel) = (768, 1024, 3)
# resized_img = cv2.resize(img, (new_y,new_x))
resized_img = cv2.resize(img, (400,300))
# plt.imshow(resized_img)
# plt.show()
# plt.imsave("orange3.jpg", resized_img)

#画像を指定範囲でクロップする
# cropped = img[y1:y2,x1:x2]
cropped = img[300:500,400:600]
# plt.imshow(cropped)
# plt.show()
# plt.imsave("orange4.jpg", cropped)

#グレースケールに変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray")
plt.show()
plt.imsave("orange5.jpg", gray)

