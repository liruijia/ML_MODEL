# -*- coding: utf-8 -*-
'''
   :param : pw_z
   :param : doc_dir
   :return : topic_coherence_list  plot
'''

from .Model import *
from .topicModel_umass import *
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
import pandas as pd
import itertools

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

class estimator():
    def __init__(self,doc_dir,model_dir,topic_list , iteration_list,argvs):
        '''argvs中的参数除去topic_num , iteration'''
        self.doc_dir = doc_dir
        self.topic_coherence = []
        self.W = argvs[0]
        self.alpha = argvs[1]
        self.beta = argvs[2]
        self.save_step = argvs[3]
        self.stop_words_path = argvs[4]
        self.dictionary_path = argvs[5]
        self.topic_list = topic_list
        self.iteration_list = iteration_list
        self.model_dir =model_dir

    def train_model_one(self,topic_num,iteration):
        model=Model(topic_num ,self.W,self.alpha,self.beta,iteration,self.save_step,
                    self.stop_words_path ,self.dictionary_path)
        model.run(self.doc_dir,self.model_dir)
        return model.pw_z

    def __plot_result_fixtopic(self,param_df):
        for topic in param_df['topic'].unique():
            to = param_df.loc[param_df['topic']==topic]
            plt.plot(x=to['iteration'], y=to['topic_coherence (u_mass)'])
        plt.title(' the group of iteration - topic_coherence (u_mass)')
        legend_list = [f'topic_{0}'.format(topic) for topic in param_df['topic'].unique()]
        plt.legend(labels=legend_list)
        plt.show()

    def __plot_result_fixiteration(self,param_df):
        for iteration in param_df['iteration'].unique():
            to = param_df.loc[param_df['topic']==iteration]
            plt.plot(x=to['topic'], y=to['topic_coherence (u_mass)'])
        plt.title(' the group of topic - topic_coherence (u_mass)')
        legend_list = [f'iteration_{0}'.format(topic) for topic in param_df['iteration'].unique()]
        plt.legend(labels=legend_list)
        plt.show()

    def estimator(self,plot_result=False):
        pa = itertools.product(self.topic_list,self.iteration_list)
        param_df = pd.DataFrame(columns=['topic','iteration','topic_coherence (u_mass)'])
        index = 0
        for io in pa :
            param_df.loc[index,['topic','iteration']]=[io[0],io[1]]
            index += 1
        for index0 ,row in param_df.iterrows():
            print('training in topic:{0} and iteration :{1}'.format(row['topic'],row['iteration']))
            pw_z=self.train_model_one(row['topic'],row['iteration'])
            umass_model = topicModel_umass()
            umass= umass_model.topic_umass(self.doc_dir,pw_z,pw_z_dir=None,with_std=False, with_support=False)
            self.topic_coherence.append(umass)
            param_df.loc[index0,'topic_coherence (u_mass)'] = umass
        if plot_result == True :
            self.__plot_result_fixtopic(param_df)
            self.__plot_result_fixiteration(param_df)

