#-*- coding:utf-8 -*-

#@Time	  :2019-9-10 & 13:06

#!@Author : eric.lucius

#!@File	  :img.py


import matplotlib.pyplot as plt
import matplotlib.image as mimage
import pylab

image=mimage.imread('lady_gaga.jpg')

print(image.shape)

# 行 nrow，列 ncol，数据总数 N
nrow=image.shape[0]
ncol=image.shape[1]

print(nrow,ncol)
print(mimage)
print("--------------------------")
#imgplot = plt.imshow(mimage)

# show a picture
#image=image.reshape(1,-1)
#-1是根据数组大小进行维度的自动推断

#若使用的是image=image.reshape成一行，分别为R一块, G块 ,B一块
# t=imgX1[222,:].reshape(3,32,32)
# print('t=  ' ,t.shape)
# image=np.transpose(t,(1,2,0))

image=image.reshape(600,428,3)

print(image.shape)

plt.imshow(image)
plt.axis('off')
plt.show()


