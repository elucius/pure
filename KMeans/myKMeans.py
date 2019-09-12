import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.cluster import KMeans
#%matplotlib inline

def handleData(image,N):
    # 归一化
    if filename[-4:]=='.jpg':
        image=image/256
        print("jpg")
    elif filename[-4:]=='.png':
        image=image[:,:,0:3]
    
    data=image.reshape((N,3))
    return data
	
# 计算欧式距离
def calDis(a,b):
    return np.sum(np.power(a-b,2))
#     return np.sum(np.fabs(a-b))

# 初始化聚类中心向量
def initCenter(data,K,seed=None):
    if seed!=None:
        random.seed(seed)
        
    m = np.shape(data)[1]
    center = np.mat(np.zeros((K, m)))
    for col in range(m):
        min_col = min(data[:, col])
        max_col = max(data[:, col])
        center[:, col] = min_col + float((max_col-min_col)) * np.random.rand(K, 1)
    return center
#     return random.sample(list(data),K)


def myKMeans(data,K,seed=None):
#   parameters check
    if K<=0:
        raise ValueError("Invalid number of initializations: K must be bigger than 0")
    if data.ndim!=2:
        raise ValueError("Invalid number of initializations: data's dimention must be 2")
    if data.shape[0]<=0 or data.shape[1]<=0:
        raise ValueError("Invalid number of initializations: data can not be empty")
    
    N=data.shape[0]
    #labels存储每个像素点对应的聚类中心向量
    labels=np.zeros(N)
    #centers存储聚类中心向量值
    centers=initCenter(data,K,seed)
#    print(centers)    //xx
    
    #损失函数值
    J=np.Inf
    dis_sum=0
    is_continue=True
    while is_continue:
        dis_sum=0
        is_continue=False

        # 计算各个像素点距离哪个聚类中心最近
        for i in range(N):
            min_dis=np.Inf
            min_center=-1
            for j in range(K):
                dis=calDis(data[i],centers[j])
                if dis<min_dis:
                    min_dis=dis
                    min_center=j
                    
            dis_sum+=min_dis
            labels[i]=min_center
            
#        print(J,dis_sum)  //xx
        
        # 若距离和不再减小，则停止
        if dis_sum<J:
            J=dis_sum
            is_continue=True   
        
        # 重新计算聚类中心
        for i in range(K):
            tmp=data[labels==i]
            if tmp.size!=0:
                centers[i]=np.mean(tmp,axis=0)
            else:
                centers[i]=random.choice(data)
    return labels,centers
    
# 生成并保存经KMeans处理过后的图像
def genKMeansImage(labels,centers,nrow,ncol,filename):
    N=nrow*ncol
    newimage=np.zeros(N*3).reshape((N,3))

    for i in np.arange(N):
        newimage[i]=kmeans.cluster_centers_[kmeans.labels_[i]]
        
    newimage=newimage.reshape((nrow,ncol,3))
    
    plt.imsave(filename,newimage,format=filename[-3:])
	
if __name__=='__main__':
    # 初始化图像，K值
    filename='lady_gaga.jpg'
    K=5
    
    # 读取图像数据
    image=mpimg.imread(filename)
    im_array = np.array(image)
    print(im_array)
    np.save("filename.npy", im_array)
    #np.savetxt("123.txt",im_array) #只能处理一二维数组

    # 行 nrow，列 ncol，数据总数 N
    nrow=image.shape[0]
    ncol=image.shape[1]
    N=nrow*ncol
    
    # 数据处理
    data=handleData(image,N)
    print(data)
    # 首先使用sklearn中的KMeans处理，作为对比
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
    # 生成并保存处理后的图像
    genKMeansImage(kmeans.labels_,kmeans.cluster_centers_,nrow,ncol
                   ,filename[:-4]+'_'+str(K)+'_sklKMeans'+filename[-4:])
    
    # 使用自己编写的myKMeans处理
    #labels,centers=myKMeans(data,K)
    # 生成并保存处理后的图像
    #genKMeansImage(labels,centers,nrow,ncol
    #              ,filename[:-4]+'_'+str(K)+'_myKMeans'+filename[-4:])