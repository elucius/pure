import numpy as np
import pandas as pd

#from sklearn.cluster import KMeans as xKMeans

data = input('请输入文件名: ')
k = int(input('分类数 ：'))
n= int(input('循环次数: '))
'''
def getdata(data):
    with open(data) as f:
        data = []
        datalist = f.readlines()
        for each in datalist:
            each = each.strip().split('\t')
            each = list(map(float,each))
            data.append(each)

        data = np.array(data)
            
        return(data)
        '''


def getdata(data):
    f = pd.read_csv(data)
    data =f.values
    truelabel = np.transpose([data[:,-1]])
    newdata = np.delete(data,-1,axis = 1)
    
    return(newdata)

def dist(A,B):
    return np.sqrt(np.sum(np.power(A - B ,2)))

def randcent(data,k):
    n = np.shape(data)[1]
    cent = np.zeros((k,n))
    for j in range(n):
        minj = min(data[:,j])
        rangej =float(max(data[:,j]) - minj)
        cent[:,j:j+1] = minj + rangej *np.random.rand(k,1)#注意此索引可以获得一个二维数组，而若只是data[:,1]获得的仅仅是一维数组
        
    return cent

   

def KMeans(data,k,n,dist=dist, creatcent=randcent):
    m = np.shape(data)[0]
    labelmat = np.zeros((m,2))

    cent = creatcent(data,k)
    num = 0
    while num<n:
        
        num +=1
        for i in range(m):
            mindist = np.inf
            minindex = -1
            for j in range(k):
                distj =dist(data[i],cent[j])
                if distj <mindist:
                    mindist = distj
                    minindex = j
            labelmat[i,:]=minindex,mindist**2
        
        for a in range(k):
            centa = data[np.nonzero(labelmat[:,0]==a)[0]]
            cent[a] = np.mean(centa,axis=0) 
    return(cent,labelmat)





data = getdata(data)
cent , labelmat = KMeans(data,k,n)


print('质心:',cent)
print('label :',labelmat)







'''
for j in range(m):
    labelmat[j,1]=dist(data[j,:],cent0)**2
 

while (len(centlist)<k):
        for i in range(len(centlist)):
            bestsse =np.inf
            newdata = data[np.nonzero(labelmat[:,0:1]==i)[0],:]

            newcent,newlabel = KMeans(newdata,2,dist)
            newsse = np.sum(newlabel[:,1:2])
            restsse=np.sum(labelmat[np.nonzero(labelmat[:,0:1] !=i)[0],1])
            if (newsse+restsse) < bestsse:
                bestsse = newsse+restsse
                bestsplit = i
                bestcent = newcent
                bestlabel = newlabel.copy()
        bestlabel[np.nonzero(bestlabel[:,0:1]==0)[0],[0]] = bestsplit
        bestlabel[np.nonzero(bestlabel[:,0:1]==1)[0],[0]] = len(centlist)
        print('最好的分点',bestsplit)
        print('label长度',len(bestlabel))
        centlist[bestsplit] = bestcent[0,:]
        centlist.append(bestcent[1,:])
        labelmat[np.nonzero(labelmat[:,0:1]==bestsplit)[0],:]=bestlabel


for i in range(len(centlist)):
    bestsse =np.inf
    newdata = data[np.nonzero(labelmat[:,0:1]==i)[0],:]
    newcent,newlabel = KMeans(newdata,2,dist)
    newsse = np.sum(newlabel[:,1:2])
    restsse=np.sum(labelmat[np.nonzero(labelmat[:,0:1] !=i)[0],1])
    if (newsse+restsse) < bestsse:
                bestsse = newsse+restsse
                bestsplit = i
                bestcent = newcent
                bestlabel = newlabel.copy()
                
bestlabel[np.nonzero(bestlabel[:,0:1]==0)[0],[0]] = len(centlist)
bestlabel[np.nonzero(bestlabel[:,0:1]==1)[0],[0]] = bestsplit
print('最好的分点',bestsplit)
print('label长度',len(bestlabel))
centlist[bestsplit] = bestcent[0,:]
centlist.append(bestcent[1,:])
labelmat[np.nonzero(labelmat[:,0:1]==bestsplit)[0],:]=bestlabel
'''
    
    

























