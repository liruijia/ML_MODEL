# -*- coding: utf-8 -*-
'''
进行模型估计的部分
    实现perplexity / u_mass 看哪一个比较好实现吧，时隔这么久 没想到又要写这个函数了  😓
topic_coherence:
   相比较更加好求一点，topic_coherence 主题一致性 其主要衡量了在同一主题下的词汇的紧密（相似程度），因此在求解topic_coherence的
   时候则找到 同一个主题下的相同的word  再在所有的语料中去遍历相似度

btm 主要考虑的是biterm 而 lda则考虑的是word
    :param   topic-word 、 corpus
    :return  u_mass
'''
import numpy as np
class topicModel_umass():
    def __init__(self):
        self.corpus= []
        self.pw_z = []
        self.D = None
        self.V = None
        self.T = None
        self.top_words = {}

    def __load_corpus(self,doc_dir):
        with open(doc_dir,'r',encoding = 'utf-8') as f:
            for line in f.readlines():
                lines = line.strip().split(' ')
                self.corpus.append(lines)
            f.close()
        self.D = self.corpus

    def __load_pw_z(self,pw_z_dir):
        with open(pw_z_dir,'r',encoding = 'utf-8') as f:
            for line in f.readlines():
                lines = list(map(float, line.strip().split(' ')))
                self.pw_z.append(lines)
        self.T =len(self.pw_z)
        self.V = len(self.pw_z[0])

    def __get_top_word_list(self,top_n=100):
        '''
        :param : top_n
        :return: top_n_list
        '''
        print(self.V)
        op = range(self.V)
        for i  in range(self.T):
            pro = self.pw_z[i]
            id_pro = list(zip(op,pro))
            word_list =  sorted(id_pro , key= lambda x : x[1] , reverse=True)[:self.T]
            top_w = [widp[0] for widp in word_list]
            self.top_words[i]=top_w

    def __get_top_word_pairs(self,topic_id):
        ''' 获取每一个topic的key_word_pair'
            :param : topic_id
            :return : top_words_pair
            只是针对于某一个topicid 求取相应的words_pair
        '''
        words_list =self.top_words[topic_id]
        n=len(words_list)
        words_pair=[]
        for i in range(n-1):
            for j in range(i+1,n):
                words_pair.append((words_list[i],words_list[j]))
        print('******* the length of the keyword of topic{0}  is {1}  \r\n'.format(topic_id , len(words_list)))
        print('******* the length of the words_pais of topic{0} is {1} \r\n'.format(topic_id , len(words_pair)))
        # 正常来说，有n个key_words 则有n(n-1)/2
        return  words_pair

    def __log_conditional_probability(self,widi ,widj , emplsion):
        '''
            widi : word_i 所对应的id
            widj : word_j 所对应的id
            emplsion : emplsion 防止出现count=0 的情况  一般来说emplsion越小越好 emplsion =1e-12
        '''
        count_j = 0
        count_ij = 0
        for doc in self.corpus:
            try :
                if widj in doc:
                    count_j += 1
                if widj in doc and widi in doc :
                    count_ij += 1
                score = np.log(((count_ij/self.D)+emplsion)/(count_j/self.D))
            except :
                score = 0.0
        return score

    def __aggreagte_segment_sims(self,score_list ,with_std,with_support):
        mean = np.mean(score_list)
        st = [mean]
        if with_std  :
            st.append(np.std(score_list))
        if with_support :
            st.append(len(score_list))
        #print('******* the final score *****\t\n')
        return st[0] if len(st)==1 else tuple(st)

    def topic_umass(self,doc_dir,pw_z,pw_z_dir=None,with_std=False, with_support=False):
        '''有一个问题，umass的计算可以是直接最后将所有topic所对应的score进行求和，也可以是经过一定的计算
           :param : with_Std  bool 是否结算std
           :param : with_support bool  是否返回score的长度
           :return : umass
        '''
        self.__load_corpus(doc_dir)
        print('pw_z',pw_z)
        if pw_z_dir !=None:
            self.__load_pw_z(pw_z_dir)
        else:
            self.pw_z= pw_z
            self.T= len(self.pw_z)
            self.V= len(self.pw_z[0])
        self.__get_top_word_list()
        topic_coherence = []
        for i in range(self.T):
            sum_score= []
            for ii , jj in self.__get_top_word_pairs(i):
                    if ii == jj :
                        continue
                    else:
                        sum_score.append(self.__log_conditional_probability(ii,jj,emplsion=0.001))
            topic_coherence.append(self.__aggreagte_segment_sims(sum_score,with_std,with_support))
        if with_std == False and with_support == False :
            return self.__aggreagte_segment_sims(topic_coherence,with_std=False,with_support=False)
        elif with_std == True :
            op = [ll[0] for ll in topic_coherence]
            return self.__aggreagte_segment_sims(op,with_std=False, with_support = False)
