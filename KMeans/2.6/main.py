#!D:\Soft\Apache24\htdocs1\v1\test\venv1\Scripts\python.exe
#coding=utf-8

import kMeans
from numpy import *
import pylab
from numpy import *

def showFigure(dataMat,k,clusterAssment):

    tag=['go','or','yo','ko']
    for i in range(k):        
        datalist = dataMat[nonzero(clusterAssment[:,0].A==i)[0]]
        pylab.plot(datalist[:,0],datalist[:,1],tag[i])
    pylab.show()


if __name__ == '__main__':
    k=4
    dataMat = mat(kMeans.loadDataSet('testSet2.txt'))
    #myCentroids,clusterAssment=kMeans.kMeans(dataMat,k)
    myCentroids,clusterAssment=kMeans.biKmeans(dataMat,k)
    print("-------------------------------------")
    showFigure(dataMat,k,clusterAssment)




