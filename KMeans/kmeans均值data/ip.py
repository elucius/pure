#!D:\Soft\Apache24\htdocs1\v1\test\venv1\Scripts\python.exe
#-*- coding:utf-8 -*-

#############ip查询############
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import dateconvert



#from sklearn.datasets import load_iris
#iris = load_iris()
#X = iris.data[:] ##表示我们只取特征空间中的后两个维度



A=np.zeros((65536,5),dtype=str)#先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
#f=open('123.txt','r',encoding='utf-8')           #打开数据文件文件
#lines=f.readlines()       #把全部数据文件读到一个列表lines中
#A_row=0                   #表示矩阵的行，从0行开始
#for line in lines:          #把lines中的数据逐行读取出来
#    list = line.strip('\n').split(' ')      #处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
#    A[A_row:] = list[0:5]                    #把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
#    A_row+=1


import pandas as pd
data = pd.read_csv("123.csv",encoding="utf-8")
X=data


#y=X[1:11].values.tolist()
#print(y)
A=X[1:65536].values.reshape(65535,5)
#A=X[1:501].values.reshape(500,5)


#将字符串形式的ip地址转成整数类型。
def ipToLong(ip_str):
    #print map(int,ip_str.split('.'))
    ip_long = 0
    for index,value in enumerate(reversed([int(x) for x in ip_str.split('.')])):
        ip_long += value<<(8*index)
    return ip_long




for i in range(65535):
    A[:,2][i]=ipToLong(str(A[:,2][i]))
    A[:, 3][i] = ipToLong(str(A[:, 3][i]))
    A[:, 1][i] = dateconvert.unix_time(str(A[:, 1][i]))


#print(A[:,2])
#print(A[:,3])
#print(A[:,1])

'''
for i in range(11):
    line=X[1:i]
    #print(line)
    xline=str(line).split(" ")
    #print(xline)

    y=X[1:i].values.tolist()
    #print(y)

'''

#绘制数据分布图
plt.scatter(A[:, 0], A[:, 1], c = "red", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()


estimator = KMeans(n_clusters=5)#构造聚类器
estimator.fit(A)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = A[label_pred == 0]
x1 = A[label_pred == 1]
x2 = A[label_pred == 2]
x3 = A[label_pred == 3]
x4 = A[label_pred == 4]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c = "yellow", marker='v', label='label2')
plt.scatter(x4[:, 0], x4[:, 1], c = "black", marker='x', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

