#!D:\Soft\Apache24\htdocs1\v1\test\venv1\Scripts\python.exe
#-*- coding:utf-8 -*-

#############ip查询############
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
import dateconvert
import pandas as pd



#from sklearn.datasets import load_iris
#iris = load_iris()
#X = iris.data[:] ##表示我们只取特征空间中的后两个维度


#初始化数据：

#f=open('123.txt','r',encoding='utf-8')           #打开数据文件文件
#lines=f.readlines()       #把全部数据文件读到一个列表lines中
#A_row=0                   #表示矩阵的行，从0行开始
#for line in lines:          #把lines中的数据逐行读取出来
#    list = line.strip('\n').split(' ')      #处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
#    A[A_row:] = list[0:5]                    #把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
#    A_row+=1



data = pd.read_csv("123.csv",encoding="utf-8")
X=data
#读取数据量，行数
n=50000
#定义分簇的组数，现在是5组
k=5


#A=X[1:65536].values.reshape(65535,5)
A=X[1:(n+1)].values.reshape(n,5)



#分析的np初始化
#np_empty= np.empty((n,3))
#for i in range(n):  # 给每一行赋值
#    np_empty[i] = i

###np_empty=np.zeros((n,3),dtype=float)#先创建一个 nx3的全零方阵A，并且数据的类型设置为float浮点型

np_empty=np.zeros((n,2),dtype=float)#先创建一个 nx2的全零方阵A，并且数据的类型设置为float浮点型

#将字符串形式的ip地址转成整数类型。
def ipToLong(ip_str):
    #print map(int,ip_str.split('.'))
    ip_long = 0
    for index,value in enumerate(reversed([int(x) for x in ip_str.split('.')])):
        ip_long += value<<(8*index)
    return ip_long



for i in range(n):
    A[:,2][i]=ipToLong(str(A[:,2][i]))
    A[:, 3][i] = ipToLong(str(A[:, 3][i]))
    A[:, 1][i] = dateconvert.unix_time(str(A[:, 1][i]))
'''
np_empty[:,0]=A[:,2]/4294967295
np_empty[:,1]=A[:,3]/4294967295
np_empty[:,2]=A[:,4]
'''

np_empty[:,0]=A[:,3]/4294967295
np_empty[:,1]=A[:,4]


#print(np_empty[:,0])
#print(np_empty[:,1])

everage1=np_empty[:,0].mean()


###everage2=np_empty[:,1].mean()


s=np.std(np_empty, axis=0)
print(s)

np_empty[:,0]=(np_empty[:,0]-everage1)/s[0]
###np_empty[:,1]=(np_empty[:,1]-everage2)/s[1]


'''
for i in range(11):
    line=X[1:i]
    #print(line)
    xline=str(line).split(" ")
    #print(xline)

    y=X[1:i].values.tolist()
    #print(y)

'''

#取用户、对方IP及端口三个维度分析
#print(A.iloc[0:2999,2:4])   #AttributeError: 'numpy.ndarray' object has no attribute 'iloc'

#数据分布图原始
#绘制数据分布图
###plt.scatter(np_empty[:, 1], np_empty[:, 2], c = "red", marker='o', label='see')
plt.scatter(np_empty[:, 0], np_empty[:, 1], c = "red", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()


estimator = KMeans(n_clusters=k)#构造聚类器
estimator.fit(np_empty)#聚类


#查询各个簇的数量
quantity = pd.Series(estimator.labels_).value_counts()
print(quantity)


#查询质心
data = np_empty
#-----------------------------------------------------------

def dist(A,B):
    return np.sqrt(np.sum(np.power(A - B ,2)))


def randcent(data, k):
    n = np.shape(data)[1]
    cent = np.zeros((k, n))
    for j in range(n):
        minj = min(data[:, j])
        rangej = float(max(data[:, j]) - minj)
        cent[:, j:j + 1] = minj + rangej * np.random.rand(k, 1)  # 注意此索引可以获得一个二维数组，而若只是data[:,1]获得的仅仅是一维数组

    return cent 

def KMeansSelf(data, k, n, dist=dist, creatcent=randcent):
    m = np.shape(data)[0]
    labelmat = np.zeros((m, 2))

    cent = creatcent(data, k)
    num = 0
    while num < n:

        num += 1

        for i in range(m):
            mindist = np.inf
            minindex = -1
            for j in range(k):
                distj = dist(data[i], cent[j])
                if distj < mindist:
                    mindist = distj
                    minindex = j
            labelmat[i, :] = minindex, mindist ** 2

        for a in range(k):
            centa = data[np.nonzero(labelmat[:, 0] == a)[0]]
            cent[a] = np.mean(centa, axis=0)
    return (cent, labelmat)

#--------------------------------------------
'''
cent, labelmat = KMeansSelf(data, k, n)


print('质心:', str(cent))
print('label :', str(labelmat))
print("--------------------------------------------------------")
'''
#--------------------------------------------


label_pred = estimator.labels_ #获取聚类标签



#绘制k-means结果
'''
x0 = A[label_pred == 0]
x1 = A[label_pred == 1]
x2 = A[label_pred == 2]
x3 = A[label_pred == 3]
x4 = A[label_pred == 4]
'''

x0 = np_empty[label_pred == 0]

x1 = np_empty[label_pred == 1]

x2 = np_empty[label_pred == 2]

x3 = np_empty[label_pred == 3]

x4 = np_empty[label_pred == 4]

#查询质心坐标
y1=estimator.cluster_centers_
print('y1=',y1)



plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c = "yellow", marker='v', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c = "black", marker='x', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

