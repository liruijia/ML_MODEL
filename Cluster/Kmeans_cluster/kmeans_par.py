# -*- coding = utf-8 -*-
'''
 @Author : RuiJia Li 
 @Time   : 2021/1/27 10:27
 @File   : kmeans_par.py
 @Desc   : 
'''
'''
input: 一组没有label的数据集
output: 数据集的簇向量存储 -- 之后训练的时候可以复用，模型预测得结果
'''

import numpy as np
import pandas as pd


class Kmeans_classify():
    def __init__(self,K,sim_type='euclidean',centroid_select_type='random'
                 ,interation_num = 200 ):
        self.K =K
        self.centroid_last = None
        self.centroid_stat = None
        self.centroid_init = None
        self.centroid_select_type = centroid_select_type
        self.sim_type = sim_type
        self.interation_num = interation_num

    def __euclidean_similar(self,x,y):
        return np.sum((x-y)**2)

    def __cosine_similar(self,x,y):
        score = np.dot(x,y)
        normal_x = np.linalg.norm(x)
        normal_y = np.linalg.norm(y)
        normal_x[normal_x==0]=1e-7
        normal_y[normal_y==0]=1e-7
        return score/(normal_x * normal_y)

    # def __random_state(self,seed):
    #     if seed is None or seed is np.random:
    #         return np.random.mtrand._rand
    #     if isinstance(seed ,  int) :
    #         return np.random.RandomState(seed)
    #     if isinstance(seed , np.random.RandomState):
    #         return seed
    #     return ValueError('{0} can\'t be used to seed'.format(seed))

    def __random_select_centroid(self,x,seed='2345'):
        n_row ,n_col = x.shape
        random_index = np.random.choice(list(range(n_row)),self.K)
        return x[random_index]

    def __most_select_centroid(self,x):
        n_row ,n_col = x.shape
        centroid = np.zeros(self.K,n_col)
        ratio = len(n_row)/self.K
        for i in range(n_col):
            sorted_col = sorted(x[:,i],reverse=False)
            value_col =[ii*ratio for ii in range(len(sorted_col)/self.K)]
            centroid[:,i] = value_col
        return centroid

    def __centroid_verify(self,x,label):
        data=pd.DataFrame(x)
        data['label']=label
        cp = data.groupby('label').mean()
        return cp.values

    def __train_random(self,x):
        if self.centroid_select_type == 'random':
            self.centroid_init = self.__random_select_centroid(x)
            self.centroid_stat = self.centroid_init
        else :
            self.centroid_init = self.__random_select_centroid(x)
            self.centroid_stat = self.centroid_init

        n_row , n_col = x.shape
        for i in range(self.interation_num):
            label = []
            for ii in range(n_row):
                if self.sim_type == 'euclidean':
                    sim_matrix = self.__euclidean_similar(x[ii,:],self.centroid_stat)
                    sim_sorted_index = np.argmax(sim_matrix)
                    label.append(sim_sorted_index)
                elif self.sim_type == 'cosine':
                    sim_matrix = self.__cosine_similar(x[ii,:],self.centroid_stat)
                    sim_sorted_index = np.argmax(sim_matrix)
                    label.append(sim_sorted_index)
            centroid_stat = self.__centroid_verify(x,label)

            # print('\r\n****centroid_stat****\r\n')
            # print(centroid_stat)
            # print('\r\n*****self.centroid_stat******\r\n')
            # print(self.centroid_stat)

            if i%10==0:
                print('the {0}-interation'.format(i))
            if np.sum(np.abs(centroid_stat-self.centroid_stat))<1e-7:
                break
            else :
                self.centroid_stat = centroid_stat
        self.centroid_last = self.centroid_stat
        np.savetxt('centroid_last.txt', self.centroid_last)
        print('*******train step  over*******')

    def __train_most(self,x):
        if self.centroid_select_type == 'most':
            self.centroid_init = self.__most_select_centroid(x)
            self.centroid_stat = self.centroid_init
        else :
            self.centroid_init = self.__most_select_centroid(x)
            self.centroid_stat = self.centroid_init

        n_row , n_col = x.shape
        for i in range(self.interation_num):
            label = []
            for ii in range(n_row):
                if self.sim_type == 'euclidean':
                    sim_matrix = self.__euclidean_similar(x[ii,:],self.centroid_stat)
                    sim_sorted_index = np.argmax(sim_matrix)
                    label.append(sim_sorted_index)
                elif self.sim_type == 'cosine':
                    sim_matrix = self.__cosine_similar(x[ii,:],self.centroid_stat)
                    sim_sorted_index = np.argmax(sim_matrix)
                    label.append(sim_sorted_index)
            centroid_stat = self.__centroid_verify(x,label)

            if i%10==0:
                print('the {0}-interation'.format(i))

            # print('\r\n****centroid_stat****\r\n')
            # print(centroid_stat)
            # print('\r\n*****self.centroid_stat******\r\n')
            # print(self.centroid_stat)

            if np.sum(np.abs(centroid_stat-self.centroid_stat))<1e-7:
                break
            else :
                self.centroid_stat = centroid_stat
        self.centroid_last = self.centroid_stat
        np.savetxt('centroid_last.txt', self.centroid_last)
        print('*******train step  over*******')

    def train(self,x):
        if self.centroid_select_type == 'random' :
            self.__train_random(x)
        else:
            self.__train_most(x)

    def predict(self,x_test):
        print('begin predit **********')
        n_row_sample = x_test.shape[0]
        centroid=np.loadtxt('./centroid_last.txt')
        label=[]
        for i in range(n_row_sample):
            if self.sim_type == 'euclidean':
                sim_matrix = self.__euclidean_similar(x_test[i,:],centroid)
                sim_sorted_index = np.argmax(sim_matrix)
                label.append(sim_sorted_index)
            elif self.sim_type == 'cosine':
                sim_matrix = self.__cosine_similar(x_test[i,:],centroid)
                sim_sorted_index = np.argmax(sim_matrix)
                label.append(sim_sorted_index)
        print('******** predict over ********')
        return label

if __name__ == '__main__':
    a=[[1.343,4.232,2.32,1.232,2.123],
       [1.2342,2.232,2.232,1.423,5.232],
       [1.232,2.34,5.232,2.232,1.242],
       [3.232,5.232,2.343,1.232,1.242],
       [1.242,1.32,4.233,1.232,1.234],
       [2.121,3.232,1.32,1.43,1.343],
       [4.23,1.232,1.12,1.45,3.121],
       [7.34,5.232,3.232]]
    b=np.array(a)
    M=Kmeans_classify(K=2,interation_num=50)
    M.train(b)
    label=M.predict(b)
    print(label)
