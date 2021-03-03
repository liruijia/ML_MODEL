# encoding = 'utf-8'
# 练习写
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from collections import Counter ,defaultdict


def get_similar_continus(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def get_dissimilar_huang(x1,x2):
    return np.sum(x1!=x2,axis=1)

def get_dissimilar_huang_cao(x1,x2,A,y2_label):
    '''
    :param x1: sample1
    :param x2: centroid x2
    :param A:  a array , which has feature+1 columns, the final column represents the cluster number of every sample
    :param y2_label: the number of sample x2
    :return: dis-similar
    '''
    n_row , n_col =A.shape
    n_feature = n_col -1
    dissim =0
    for a in range(n_feature) :
        if x1[a] != x2[a]:
            dissim +=1
        else:
            cl = np.sum(A[:,-1]==y2_label)
            cl_index  = A[:,-1] == y2_label
            cla = np.sum(A[cl_index, a]==x[a])
            dissim += 1-cla/cl
    return dissim

def find_proto_column(data,K):
    continus_column=[]
    object_column = []
    num = random.sample(range(len(data)),K)
    # for i in range(len(data[0,:])):
    #     if isinstance(data[0,i],int)  or isinstance(data[0,i],float) :
    #         continus_column.append(i)
    #     elif isinstance(data[0,i],str):
    #         object_column.append(i)
    #     else :
    #         raise ValueError ('the value-type of x[{0}] is error'.format(i))


    for  i in range(len(data[0,:])):
        if isinstance(data[0,i],float):
            continus_column.append(i)
        elif isinstance(data[0,i],str):
            object_column.append(i)
        elif isinstance(data[0,i],int):
            if np.unique(data[:,i])<=20:
                object_column.append(i)
            else :
                continus_column.append(i)
        else :
            raise ValueError ('the value-type of x[{0}] is error'.format(i))

    continus_data = data[:,continus_column]
    object_data = data[:,object_column]

    continus_proto = continus_data[num,:]
    object_proto = object_data[num,:]

    return continus_column , object_column , continus_data ,object_data,continus_proto ,object_proto

def random_state(seed):
    # 返回一个随机器 randomstate
    if seed is None or seed is np.random :
        return np.random.mtrand._rand
    if isinstance(seed ,  int) :
        return np.random.RandomState(seed)
    if isinstance(seed , np.random.RandomState):
        return seed
    return ValueError('{0} can\'t be used to seed'.format(seed))

def  init_centroids_huang(X, n_cluster, dissim ,random_state):
    'huang  k-prototype 模型主要针对的是 离散型数据，因此 其所求到的 簇中心也是 离散数据，\
    huang 对于每一列的数据进行统计，然后从统计到的属性值中随机生成n_cluster即可'
    n_col = X.shape[1]
    centroids = np.zeros((n_cluster , n_col))
    for i in range(n_col):
        ferq = defaultdict(int)
        for coi in  X[:,i]:
            ferq[coi]+=1
        nop = [ui  for ui , value in ferq.items()]
        choice_num = sorted(nop)
        centroids[:,i] = random_state.choice(choice_num , n_cluster)
    for idx  in n_cluster :
        ndx = np.argsort(dissim(X,centroids[idx]))
        while np.all(X[ndx[0]] == centroids, axis=1).any() and ndx.shape[0] > 1:
            ndx = np.delete(ndx, 0)
        centroids[idx] = X[ndx[0]]
    return centroids

def init_centroids_cao(X,n_cluster, dissim):
    n_point , n_attr = X.shape[0]
    dens = np.zeros(n_point)
    centroids = np.zeros((n_cluster, n_attr))
    for att in  range(n_attr):
        freq = defaultdict(int)
        for i in X[:,att]:
            freq[i]+=1
        for point in range(n_point):
            dens[point] += freq[X[point,att]]/n_attr/n_point
    centroids[0] = X[np.argmax(dens)]
    if n_cluster >1 :
        for ik in range(1,n_cluster):
            dis = np.empty((ik, n_point))
            for ikk in  range(ik):
                dis[ikk] = dissim(X,centroids[ikk])*dens
            centroids = X[np.argmax(np.min(dis,axis=0))]
    return centroids

def evaluation_index(true_label , predict_label):
    n_sample = len(true_label)
    K = len(np.unique(true_label))
    A_freq = defaultdict(int)
    B_freq = defaultdict(int)
    C_freq = defaultdict(int)
    PR ,RE , AC = 0,0 ,0
    for i in range(K):
        for ji in range(n_sample):
            if true_label[ji] == i and predict_label[ji]==i:
                A_freq[i]+=1
            if true_label[ji] != i and predict_label[ji] == i :
                B_freq[i]+=1
            if true_label[ji] == i and predict_label[ji] != i :
                C_freq[i]+=1
    for i in range(K):
        pri = A_freq[i] / (A_freq[i] + B_freq[i])
        rei = A_freq[i] / (A_freq[i] + C_freq[i])
        aci = A_freq[i]
        PR  += pri/K
        RE  += rei/K
        AC  += aci/n_sample
    print('\t the precision of clustring is {0} \r\n'.format(PR))
    print('\t the recall of clustering  is {0} \r\n'.format(RE))
    print('\t the accuracy_score of clusting is {0} \r\n'.format(AC))
    return PR,RE,AC

def kprototype_cluster(data,K,max_iter):
    '''
    :param data:
    :param K:
    :param max_iter :
    :param type: type给出来了初始化变量的方式
    :return:  返回所有样本的簇列表
    '''
    row = len(data)
    col = len(data[0])

    continus_column,object_column ,continus_data,object_data ,continus_proto , object_proto = find_proto_column(data,K)

    cluster_list = []
    cluster_count = {}

    sumcluster_continus = {}
    sumcluster_object = {}
    for i  in range(row):
        min_distance = float('inf')
        cluster = i
        for j in range(K):
            distance = get_similar_continus(continus_data[i,:], continus_proto[j,:])+\
                       get_dissimilar_huang(object_data[i,:],object_proto[j,:])
            if distance <= min_distance :
                min_distance = distance
                cluster = j
        cluster_list.append(cluster)

        if cluster_count.get(cluster)==None :
            cluster_count[cluster] =1
        else:
            cluster_count[cluster]+=1

        for ipt in range(len(continus_proto[0,:])):
            if sumcluster_continus.get(cluster) ==None :
                sumcluster_continus[cluster] = [continus_data[i,ipt]] + [0] * (len(continus_column) - 1)
            else:
                sumcluster_continus[cluster][ipt] += continus_data[cluster][ipt]
            continus_proto[cluster][ipt] = sumcluster_continus[cluster][ipt]/cluster_count[cluster]

        for iou in range(len(object_proto[0,:])):
            if sumcluster_object.get(cluster) == None :
                sumcluster_object[cluster] = [Counter(object_data[i,j])] + [Counter()] * (len(object_column) - 1)

            else:
                sumcluster_object[cluster][j] += Counter(object_data[i,j])
            object_proto[cluster,j] = sumcluster_object[cluster][j].most_common()[0][0]

    for m in range(max_iter):
        for i  in range(row):
            min_distance = float('inf')
            cluster = i
            for j in range(K):
                distance = get_similar_continus(continus_data[i,:], continus_proto[j,:])+ \
                           get_dissimilar_huang(object_data[i,:],object_proto[j,:])
                if distance < min_distance :
                    min_distance = distance
                    cluster = j

            if cluster_list[i] != cluster :
                cluster_list[i] =cluster
                cluster_count[cluster] +=1
                cluster_count[cluster_list[i]] -= 1

            for ipt in range(len(continus_proto[0,:])):
                sumcluster_continus[cluster][ipt] += continus_data[cluster][ipt]
                sumcluster_continus[cluster_list[i]][ipt] -=continus_data[cluster][ipt]
                continus_proto[cluster][ipt] = sumcluster_continus[cluster][ipt] / cluster_count[cluster]
                continus_proto[cluster_list[i]][ipt] = sumcluster_continus[cluster_list[i]][ipt] / cluster_count[cluster]

            for ipt in range(len(object_proto[0,:])):
                sumcluster_object[cluster][ipt] += object_data[cluster][ipt]
                sumcluster_object[cluster_list[i]][ipt] -= object_data[cluster][ipt]
                object_proto[cluster,j] = sumcluster_object[cluster][j].most_common()[0][0]
                object_proto[cluster_list[i],j] = sumcluster_object[cluster_list[i]][j].most_common()[0][0]

        if (m+1)%10==0:
            print('the {0}/{1} iter train'.format(m+1,max_iter))
    return cluster_list

if __name__ =='__main__':
    train_data = load_iris()
    x_train , y_train = train_data.data , train_data.target
    m,n = x_train.shape
    x_bc = np.array(np.random.choice(20,(m,3)),dtype = np.int)
    x_final = np.hstack((x_train , x_bc))

    continus_column , object_column ,continus_data , object_data  ,continus_proto , object_proto = find_proto_column(x_final , 4)


