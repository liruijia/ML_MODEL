#encoding : utf-8

import pandas as pd
import re
from zhon.hanzi import punctuation
import jieba
from gensim.models import word2vec ,LdaModel , CoherenceModel
from  gensim import corpora
import numpy as np
from sklearn.cluster import  KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score ,calinski_harabasz_score
import itertools

class handle_data():
    def __init__(self):
        self.data_train = None
        self.data_cut_text = None
        self.data_matrix = None
        self.stop_words = None
        self.data_text_sentence = None
        self.data_corpus = None
        self.data_dictionary = None
        self.vocabulary_list_total = None  #经过切分之后所包含的所有的词汇集合

    def __sentence_map(feedback_code):
        if feedback_code == 1:
            return '缴费失败'
        elif feedback_code == 2:
            return '担心资金安全'
        elif feedback_code == 3:
            return '更换住处'
        elif feedback_code == 4:
            return '其他'
        elif feedback_code == 5 :
            return '手误签约'
        elif feedback_code == 6:
            return '缴费重复'

    def __loaddata(self,path):
        print('*** load data ***')
        da = pd.read_csv(path)
        map_x = {1:'缴费失败',2:'担心资金安全',3:'更换住处',4:'其他',5:'手误签约',6:'缴费重复'}
        da['feedback_mess_total']=['feedback_msg']
        da.loc[da['feedback_msg']=='missing','feedback_mess_total']=da.loc[da['feedback_msg']=='missing','feedback_code'].map(map_x)
        for index, row in da.iterrows():
            if str(row['feedback_mess_total']).isalnum():
                da.loc[index,'feedback_mess_total']=self.__sentence_map(row['feedback_code'])
            elif re.match(r'[。* |，*|？.*|\t*]',row['feedback_mess_total']) is not None :
                da.loc[index,'feedback_mess_total'] = self.__sentence_map(row['feedback_code'])
            elif re.match(r'[^gnoOGN][\?\，\。]\w+',row['feedback_mess_total']) is not None:
                da.loc[index,'feedback_mess_total'] = self.__sentence_map(row['feedback_code'])
            elif len(row['feedback_mess_total'])==1:
                da.loc[index,'feedback_mess_total'] = self.__sentence_map(row['feedback_code'])
            elif row['feedback_mess_total']=='g?h' :
                da.loc[index,'feedback_mess_total'] = self.__sentence_map(row['feedback_code'])
        self.data_train = da
        print('**** over handle ****')

    def __load_stopwords(self,stop_words_path):
        print('**** load stop_words ****')
        stop_words = []
        with open(stop_words_path , 'r', encoding = 'utf-8') as f:
            for line in f:
                stop_words.append(line)
            f.close()
        stop_words.append(['元','说','了','多元','的','从','啊','10','30','100','300','扣','度','我','换了','地区'
                              ,'块','块钱','佳木斯市','前进区','会','时候','后','由','要','交','号','想','等','赤峰',''])
        self.stop_words = stop_words

    def handledata(self,data_path , stop_words_path):
        self.__loaddata(data_path)
        self.__load_stopwords(stop_words_path)
        cut_text=[]
        vocabulary_list = []
        text_words = self.data_train['feedback_mess_total']
        for sentence in text_words:
            new_se=re.sub(r'[%s,\t,\\]+'%punctuation,' ',sentence)
            cut_line =  jieba.cut(new_se)
            temp_se=[]
            for word in cut_line :
                if word  not in self.stop_words :
                    if word != '' and word.isalnum() is True and len(word)!=1:
                        if word.isalnum() :
                            word = word.lower()
                        temp_se.append(word)
                        vocabulary_list.append(word)
            cut_text.append(' '.join(temp_se))
        self.data_cut_text=cut_text
        self.vocabulary_list_total = vocabulary_list

    def handle_word2vec_data(self,data_path,stop_words_path):
        self.handledata(data_path,stop_words_path)
        text_sentence = []
        for se in self.data_cut_text :
            ui = list(se.split(' '))
            text_sentence.append(ui)
        self.data_text_sentence =text_sentence


    def handle_kmeans_data(self,data_path,stop_words_path,n_size=100,min_count=1,window=5):
        self.handle_word2vec_data(data_path,stop_words_path)
        model=word2vec.Word2Vec(self.data_text_sentence,size=n_size,window=window,min_count=min_count)
        word_list_model = model.wv.index2word

        self.N= len(self.data_text_sentence)
        dk_text_matrix = np.zeros((self.N,n_size))
        for i , sentence in enumerate(self.data_cut_text) :
            ui=list(sentence.split(' '))
            for word in ui:
                dk_text_matrix[i,:] += model.wv.get_vector(word)
        self.data_matrix = dk_text_matrix

    def handle_lda_data(self,data_path,stop_words_path ):
        self.handle_word2vec_data(data_path,stop_words_path)
        dictionary = corpora.Dictionary(self.data_text_sentence)
        corpus = [dictionary.doc2bow(text) for text in self.data_text_sentence]
        self.data_corpus = corpus
        self.data_dictionary = dictionary

class train():
    def __init__(self,data_text,data_matrix,corpus,lda_text,dictionary,lda_alpha,num_topic):
        #
        self.data_text= data_text
        # lda
        self.corpus = corpus
        self.lda_text = lda_text
        self.dictionary = dictionary
        self.data_matrix = data_matrix
        self.alpha=lda_alpha
        self.lda_num_topic =  num_topic

        self.kmeans_ch_score = None
        self.kmeans_silhouette_score = None
        self.kmeans_model =None

        self.lda_topic_coherence = []
        self.lda_model = None


    def __kmeans_result_plot(self,labels_predict_kmeans):
        mark_list = ['_','o','v','d','x','^','2','1','3','4','.','*',',','h','s','p','H','+','D','|']
        for i in range(len(self.data_matrix)):
            plt.scatter(self.data_matrix[i,0],self.data_matrix[i,5],marker=mark_list[labels_predict_kmeans[i]])
        plt.title('the  cluster group  of corpus ')
        plt.show()

    def __kemans_result_save(self,save_result_path,predict_kmeans):
        with open(save_result_path , 'w') as f :
            f.write('sentence'+','+'label_kmeans')
            f.write('\t\n')
            for i in range(len(self.data_matrix)):
                f.write(self.data_text[i] + ',' + str(predict_kmeans[i]))
                f.write('\t\n')
            f.close()

    def __kmeans_model_one(self,n_clusters):
        kmeans_model = KMeans(n_clusters= n_clusters)
        kmeans_model.fit(self.data_matrix)
        labels_predict_kmeans = kmeans_model.predict(self.data_matrix)
        return labels_predict_kmeans

    def __kmeans_choice_param(self,result_df):
        best_cluster = 0
        best_score = result_df.loc[0,'CH_SCORE']
        for index in range(1,len(result_df)):
            if  abs( result_df.loc[index , 'CH_SCORE'] , best_score ) <= 1e-4:
                best_cluster = result_df.loc[index , 'cluster']
                best_score = result_df.loc[index , 'CH_SCORE']
        return best_cluster , best_score

    def __kmeans_model_tuning(self,cluster_list):
        labels_predict_kmeans_tuning  = []
        CH_score = []
        Silhouette_score = []
        for  n_cluster in cluster_list :
            pre = self.__kmeans_model_one(n_cluster)
            labels_predict_kmeans_tuning.append(pre)
            # print(type(labels_predict_kmeans_tuning))
            # print(type(dk_matrix))
            ch = calinski_harabasz_score(self.data_matrix,pre)
            ss = silhouette_score(self.data_matrix, pre)
            print(ch , ss)
            CH_score.append(ch)
            Silhouette_score.append(ss)
        self.kmeans_ch_score = CH_score
        self.kmeans_silhouette_score =Silhouette_score
        result_df = pd.DataFrame(data={'cluster': cluster_list , 'CH_SCORE':CH_score})
        best_cluster , best_score = self.__kmeans_choice_param(result_df)
        return best_score , best_cluster

    def kmeans_train(self,clusters_list,save_path=None, if_plot = False,if_save_result = False):
        best_score , best_cluster = self.__kmeans_model_tuning(clusters_list)
        kmean_model = KMeans(n_clusters=best_cluster)
        kmean_model.fit(self.data_matrix)
        self.kmeans_model = kmean_model
        pre = kmean_model.labels_
        ch = calinski_harabasz_score(self.data_matrix,pre)
        ss = silhouette_score(self.data_matrix, pre)
        if if_save_result == True :
            self.__kemans_result_save(save_path, pre)
        if if_plot == True :
            self.__kmeans_result_plot(pre)
        print('the  score of kmeans : \t\n')
        print('calinski_harabasz_score : {0} \t\n'.format(ch))
        print('the silhouette_score : {0} \t\n'.format(ss))

    #  开始写lda

    def __lda_train_tuning(self, iteration_list):

        params_df = pd.DataFrame(columns = ['iteration', 'topic_coherence (u_mass)'])
        params_df['iteration']=iteration_list
        for index ,row in params_df.iterrows():
            lda_model = LdaModel(corpus=self.corpus,num_topics=self.lda_num_topic,
                                 id2word=self.dictionary,
                                 iterations=row['iteration'] , alpha = self.alpha)
            topic_coherence = CoherenceModel(model=lda_model,topics=self.lda_num_topic,
                                             corpus=self.corpus,
                                             dictionary=self.dictionary,coherence = 'c_v',
                                             texts =self.lda_text)
            self.lda_topic_coherence.append(topic_coherence)
            params_df.loc[index,'topic_coherence (u_mass)'] = topic_coherence
        best_iteration=params_df.loc[params_df['topic_coherence (u_mass)']==min(params_df['topic_coherence (u_mass)'])]['iteration']
        return  best_iteration

    def __lda_plot_param(self):
        # self.lda_topic_coherence
        return

    def lda_train(self,iteration_list, save_path = None ,
                  if_save_result = False,if_plot_params = False):
        best_iteration = self.__lda_train_tuning(iteration_list)
        lda_model = LdaModel(corpus=self.corpus,id2word=self.dictionary ,
                             num_topics= self.lda_num_topic
                             ,iterations=best_iteration,alpha=self.alpha)

        self.lda_model =lda_model

    def lda_predict_all(self,new_document):
        # 获取到语料库中每一则评论的主题
        # new_document 的格式为 划分好词的列表 ['','','']
        labels_predict_lda = []
        model_lda = self.lda_model
        new_bow =model_lda.id2word.doc2bow(new_document)
        all_topics = model_lda.get_document_topics(new_bow, per_word_topics=True) # 获取了整个语料库的文档主题
        cnt = 0
        for doc_topics, word_topics, phi_values in all_topics[:10]:
            print('新文档:{} \n'.format(cnt), self.lda_text[cnt])
            doc_topics = [(i[0],i[1]) for i in doc_topics]  #文档在每一个主题下的概率
            word_topics = [(self.dictionary.id2token [i[0]],i[1]) for i in word_topics]
            phi_values = [(self.dictionary.id2token [i[0]],i[1]) for i in phi_values ]
            print('文档主题:', doc_topics)
            print('词汇主题:', word_topics)
            print('Phi值:', phi_values)
            print(" ")
            print('-------------- \n')
            cnt+=1

    def dk_result_count(self):
        return
