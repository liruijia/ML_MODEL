import os
import pandas  as pd
import numpy as np
import random
import jieba
from wordcloud import WordCloud,ImageColorGenerator
from collections import Counter
import seaborn as sns
from pylab import mpl
from xeger import Xeger

from gensim.models import word2vec
from sklearn.cluster import KMeans
from gensim.models import LdaModel ,TfidfModel,CoherenceModel
from gensim import corpora
import re
from zhon.hanzi import punctuation
import time
import itertools
import collections as Counter   
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyLDAvis.gensim
import warnings
from sklearn.metrics import calinski_harabasz_score , silhouette_score


mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

abs_input_path='activity_relate/dk_text_analysis/base_cluster/input_data'
abs_output_path = 'activity_relate/dk_text_analysis/base_cluster/output_data'
jieba.load_userdict(abs_input_path+'/代扣词典.txt')
path = abs_input_path+'/train.csv'
train = pd.read_csv(path,engine='python')
print(train.head())


map_x = {1:'缴费失败',2:'担心资金安全',3:'更换住处',4:'其他',5:'手误签约',6:'缴费重复'}
train['feedback_mess_total']=train['feedback_msg']
train.loc[train['feedback_msg']=='missing','feedback_mess_total']=train.loc[train['feedback_msg']=="missing",
                                                                       'feedback_code'].map(map_x)

def sentence_map(feedback_code):
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
for index, row in train.iterrows():
    if str(row['feedback_mess_total']).isalnum():
        train.loc[index,'feedback_mess_total']=sentence_map(row['feedback_code'])
    elif re.match(r'[。* |，*|？.*|\t*]',row['feedback_mess_total']) is not None :
        train.loc[index,'feedback_mess_total'] = sentence_map(row['feedback_code'])
    elif re.match(r'[^gnoOGN][\?\，\。]\w+',row['feedback_mess_total']) is not None:
        train.loc[index,'feedback_mess_total'] = sentence_map(row['feedback_code'])
    elif len(row['feedback_mess_total'])==1:
        train.loc[index,'feedback_mess_total'] = sentence_map(row['feedback_code'])
    elif row['feedback_mess_total']=='g?h' :
        train.loc[index,'feedback_mess_total'] = sentence_map(row['feedback_code'])

text_words_else = train.loc[train['feedback_mess_total']=='其他',['msg_id','feedback_mess_total']]
text_words_else0 = text_words_else['feedback_mess_total']
text_words_noelse = train.loc[train['feedback_mess_total']!='其他',['msg_id','feedback_mess_total']]
text_words_noelse0 = text_words_noelse['feedback_mess_total']
text_words = text_words_noelse0.values
n_nums = len(text_words)

# 开始进行分词
stop_path =abs_input_path+'/stop_words.txt'
stop_words = []
with open(stop_path , 'r', encoding = 'utf-8') as f:
    for line in f:
        stop_words.append(line)
    f.close()
stop_words.append(['元','说','了','多元','的','从','啊','10','30','100','300','扣','度','我','换了','地区'
                   ,'块','块钱','佳木斯市','前进区','会','时候','后','由','要','交','号','想','等','赤峰',
                   ])
cut_text=[]
vocabulary_list = []


def get_cut_text(textwords):
    text_sentence=[]
    cut_text=[]
    for index,sentence in enumerate(textwords):
        new_se=re.sub(r'[%s,\t,\\]+'%punctuation,' ',sentence)
        cut_line =  jieba.cut(new_se.strip())
        temp_se=[]
        for word in cut_line :
            if word  not in stop_words :
                if word != '' and word.isalnum() is True and len(word)!=1:
                    if word.isalnum() :
                        word = word.lower()
                    temp_se.append(word)
                    vocabulary_list.append(word)
        cut_text.append(' '.join(temp_se))
    for se in cut_text :
        ui = list(se.split(' '))
        text_sentence.append(ui)
    dictionary = corpora.Dictionary(text_sentence)
    corpus = [dictionary.doc2bow(text) for text in text_sentence]
    return text_sentence,cut_text,vocabulary_list, dictionary ,corpus

text_sentence_noelse,cut_text_noelse,vocabulary_list_noelse,dictionary_noelse,corpus_noelse = get_cut_text(text_words_noelse0.values)

# 导出切分好的数据
save_cuttext_path = abs_output_path + '/cut_text.txt'
with open(save_cuttext_path , 'w' ,encoding = 'utf-8') as f :
    f.write('the origin text '+' '*5+','+' '*5+' the cut text ')
    f.write('\t\n')
    for index, sentence in enumerate(text_sentence_noelse):
        f.write(text_words_noelse0.values[index]+' '*5+','+' '*5+' '.join(sentence))
        f.write('\t\n')
    f.close()

# 统计这些词汇的词频，最后引入到word2cloud模型中

voca_count ={}

for line in cut_text_noelse :
    lines=line.strip().split(' ')
    for word in lines :
        voca_count[word]=voca_count.get(word,0)+1


#-- 词频统计 & 词云图 & 词频统计图
def get_wordfrequence_and_word2cloud(cut_text,vocabulary_list):
    sentence_text = ' '.join(cut_text)
    word_count = {}
    for word in vocabulary_list :
        word_count[word]=word_count.get(word,0)+1
    word_count_rank =  dict(sorted(word_count.items(), key=lambda x : x[1] ,reverse=True)[:50])
    word_count_rank_df = pd.DataFrame(data={'word_list':list(word_count_rank.keys()),
                                            'word_num':list(word_count_rank.values())})
    plt.bar(x='word_list',height='word_num',data=word_count_rank_df)
    plt.xticks(rotation=45)
    plt.show()

    model=WordCloud(background_color='white',min_font_size=5,font_path='msyh.ttc',max_words=3000
                    ,max_font_size=1000)
    model.generate(text=sentence_text )
    plt.figure("词云图")
    plt.imshow(model)
    plt.axis("off")
    plt.show()

get_wordfrequence_and_word2cloud(cut_text_noelse,vocabulary_list_noelse)


def get_word2vec_matrix(text_sentence,cut_text,n_row,n_col=100):
    model=word2vec.Word2Vec(text_sentence,size=n_col,window=5,min_count=1)
    word_list_model = model.wv.index2word
    # 整个句子的词向量
    dk_text_matrix = np.zeros((n_row,n_col))
    for i , sentence in enumerate(cut_text) :
        ui=list(sentence.split(' '))
        for word in ui:
            dk_text_matrix[i,:] += model.wv.get_vector(word)
    # 每个单词的词向量矩阵
    vocabulary_word_matrix = np.zeros((len(word_list_model),n_col))
    for i , word in enumerate(word_list_model):
        vocabulary_word_matrix[i,:]=model.wv.get_vector(word)
    return dk_text_matrix,vocabulary_word_matrix,word_list_model

dk_text_matrix_noelse,vocabulary_word_matrix_noelse,word_list_model_noelse=get_word2vec_matrix(text_sentence_noelse,cut_text_noelse,n_nums,n_col=100)

# 对语料进行聚类分析
n_clusters = 20
def get_kmeans(dk_matrix,n_clusters):
    kmeans_model = KMeans(n_clusters= n_clusters)
    kmeans_model.fit(dk_matrix)
    labels_predict_kmeans = kmeans_model.labels_
    return labels_predict_kmeans

n_clusters_list = list(range(2,60,2))

def get_kmeans_model_tuning(dk_matrix , clusters_list):
    labels_predict_kmeans_tuning  = []
    CH_score = []
    Silhouette_score = []
    for  n_cluster in clusters_list :
        pre = get_kmeans(dk_matrix,n_cluster)
        labels_predict_kmeans_tuning.append(pre)
        # print(type(labels_predict_kmeans_tuning))
        # print(type(dk_matrix))
        ch = calinski_harabasz_score(dk_matrix,pre)
        ss = silhouette_score(dk_matrix  , pre)
        print(ch , ss)
        CH_score.append(ch)
        Silhouette_score.append(ss)
    return  CH_score , Silhouette_score , labels_predict_kmeans_tuning

CH_score , Silhouette_score ,label_predict_kmeans_all_noelse= get_kmeans_model_tuning(dk_text_matrix_noelse , n_clusters_list)

ch_score_final, silhouette_score_final,label_predict_kmeans_final_noelse= get_kmeans_model_tuning(dk_text_matrix_noelse ,[40])

print(ch_score_final , silhouette_score_final)
## 结果展示图

plt.plot(n_clusters_list, CH_score)
plt.xlabel('clusters')
plt.ylabel('ch_score')
plt.show()

plt.plot(n_clusters_list , Silhouette_score)
plt.xlabel('clusters')
plt.ylabel('silhou_score')
plt.show()
## 将最终调完参的最优的结果保存
kmeans_result_path = abs_output_path + '/kmeans_result_part.csv'
with open(kmeans_result_path , 'w') as f :
    f.write('sentence'+','+'label_kmeans')
    f.write('\t\n')
    for i in range(len(dk_text_matrix_noelse)):
        f.write(cut_text_noelse[i] + ',' + str(label_predict_kmeans_final_noelse[0][i]))
        f.write('\t\n')
    f.close()
##


cluster_dk  = [[] for _ in range(40)]
for  i ,  label in enumerate(label_predict_kmeans_final_noelse[0]) :
    cluster_dk[label].append(cut_text_noelse[i])

# 使用tf-idf 、 ldamodel 阶段的数据处理
def get_tfidf(corpus):
    tfidf_model = TfidfModel(corpus)
    tfidf_matrix = []
    for tfidf in tfidf_model[corpus]:
        tfidf_matrix.append(tfidf)
    sentence_keyword_df = pd.DataFrame(columns = ['sentence','keyword','keyword_tfidf'])
    return tfidf_matrix , sentence_keyword_df

tfidf_matrix_noelse,sentence_keyword_df_noelse = get_tfidf(corpus_noelse)


matrix_row =len(corpus_noelse)
matrix_col = 0
for index in dictionary_noelse:
    if index >= matrix_col:
        matrix_col = index
matrix_col = matrix_col+1
matrix_tfidf = np.zeros((matrix_row,matrix_col))
for index,po in enumerate(tfidf_matrix_noelse) :
    for ui in po:
        id ,tfidf = ui[0],ui[1]
        matrix_tfidf[index,id]=tfidf

tfidf_clusters = 30
tf_idf_kmeans = KMeans(n_clusters = tfidf_clusters)
tf_idf_kmeans.fit(matrix_tfidf)
tfidf_kmeans_predict = tf_idf_kmeans.labels_


word2vec_kmeans_predict0 = text_words_noelse
word2vec_kmeans_predict0['cluster_result']=label_predict_kmeans_final_noelse[0]
word2vec_result_noelse =  word2vec_kmeans_predict0[['feedback_mess_total','cluster_result']]
word2vec_kmeans_predict1 = text_words_else
word2vec_kmeans_predict1['cluster_result'] =['{0}_其他'.format(tfidf_clusters) for _ in range(len(text_words_else))]
word2vec_result_else=word2vec_kmeans_predict1[['feedback_mess_total','cluster_result']]
word2vec_result_final = pd.concat([word2vec_result_noelse,word2vec_result_else])
word2vec_result_final['id']=word2vec_result_final.index.values
word2vec_dk_henan_result = word2vec_result_final.groupby(by='cluster_result').count()



# tfidf
tfidf_kmeans_predict0 =text_words_noelse
tfidf_kmeans_predict0['cluster_result'] = tfidf_kmeans_predict
tfidf_result_noelse = tfidf_kmeans_predict0[['feedback_mess_total','cluster_result']]
tfidf_kmeans_predict1 = text_words_else
tfidf_kmeans_predict1['cluster_result'] = ['{0}_其他'.format(tfidf_clusters) for _ in range(len(text_words_else))]
tfidf_result_else = tfidf_kmeans_predict1[['feedback_mess_total','cluster_result']]
tfidf_result_final = pd.concat([tfidf_result_noelse , tfidf_result_else])
tfidf_result_final['id'] = tfidf_result_final.index.values
tfidf_dk_henan_result = tfidf_result_final.groupby(by='cluster_result').count()


path_tfidf_kmeans = abs_output_path + '/tfidf_kmeans_result_part.csv'
with open(path_tfidf_kmeans, 'w') as f :
    f.write('sentence'+','+'tfidf_kmeans_result')
    f.write('\t\n')
    for i  in range(len(corpus_noelse)):
        sentence = corpus_noelse[i]
        sentence_cp=[]
        for wi in sentence :
            id, co = wi[0],wi[1]
            wo = dictionary_noelse[id]
            sentence_cp.append(wo)
        sentence_cp = ' '.join(sentence_cp)
        label = tfidf_kmeans_predict[i]
        f.write(sentence_cp+','+str(label))
        f.write('\t\n')
    f.close()

path_kmeans_vs = abs_output_path + '/vs_kmeans_result_part.csv'
with  open(path_kmeans_vs  , 'w') as f :
    f.write('sentence'+','+'tfidf_kmeans_result'+','+'word2vec_kmenas_result')
    f.write('\t\n')
    for i in range(len(corpus_noelse)):
        sentence = corpus_noelse[i]
        sentence_cp=[]
        for wi in sentence :
            id, co = wi[0],wi[1]
            wo = dictionary_noelse[id]
            sentence_cp.append(wo)
        sentence_cp = ' '.join(sentence_cp)
        tfidf = tfidf_kmeans_predict[i]
        word2vec=label_predict_kmeans_final_noelse[0][i]
        f.write(sentence_cp+','+str(tfidf)+','+str(word2vec))
        f.write('\t\n')
    f.close()


keyword_path=abs_output_path+'/keyword_result.csv'
ii=0
with open(keyword_path , 'w' ) as f:
    f.write('sentence'+','+'keyword'+','+'keyword_tfidf')
    f.write('\t\n')
    for sentence_tfidf in tfidf_matrix_noelse:
        sentence = []
        for word_tfidf in sentence_tfidf :
            word=word_tfidf[0]
            sentence.append(dictionary_noelse[word])
        sentencecp=' '.join(sentence)
        rank_sentence = sorted(sentence_tfidf,key= lambda x :x[1],reverse=True)[0]
        keyword = dictionary_noelse[rank_sentence[0]]
        keyword_tfidf = rank_sentence[1]
        sentence_keyword_df_noelse.loc[ii,:]=[sentencecp,keyword,keyword_tfidf]
        ii+=1
        f.write(sentencecp+','+keyword+','+str(keyword_tfidf))
        f.write('\t\n')
    f.close()

# 对于这些关键字进行词云展示
keyword_list = ' '.join(list(sentence_keyword_df_noelse['keyword'].values))
model=WordCloud(background_color='white',min_font_size=5,font_path='msyh.ttc',max_words=3000
                ,max_font_size=1000)
model.generate(text=keyword_list)
plt.figure("词云图")
plt.imshow(model)
plt.axis("off")
plt.show()

model_lda =LdaModel(corpus=corpus_noelse,num_topics=22,id2word=dictionary_noelse,alpha=0.01,iterations=150)
print(model_lda.print_topic(topicno=10,topn=5))

labels_predict_lda_noelse = []
for i in range(len(corpus_noelse)):
    topic,prob = model_lda.get_document_topics(bow=corpus_noelse[i])[0][0],model_lda.get_document_topics(bow=corpus_noelse[i])[0][1]
    labels_predict_lda_noelse.append(topic)
    
# 设置多个参数  求取笛卡尔积
iteration_list = list(range(10,330,20)) # 16个参数
topic_list = list(range(18,34,2))
dikaer_parm = itertools.product(iteration_list , topic_list )
params_df = pd.DataFrame(columns = ['iteration','topic', 'topic_coherence (u_mass)'])
ii_num = 0
for i in dikaer_parm:
    params_df.loc[ii_num , ['iteration' , 'topic']] = [i[0],i[1]]
    ii_num +=1

# lda 模型调优
def get_model_tuning(corpus_text,dictionary_text,text_sentence,param_result):
    start_time = time.time()
    params = param_result.copy()
    for index , row in  params.iterrows():
        print('the {0} times train '.format(index))
        lda_model = LdaModel(corpus=corpus_text , num_topics = row['topic'],alpha =0.01 ,
                             id2word=dictionary_text ,iterations= row['iteration'])
        coherence =CoherenceModel(model=lda_model, texts=text_sentence,corpus=corpus_text,
                                  dictionary=dictionary_text,coherence= 'u_mass')
        print(coherence.get_coherence())
        params.loc[index, 'topic_coherence (u_mass)'] = coherence.get_coherence()
    end_time = time.time()
    print('the total in train model : {0}'.format(end_time - start_time))
    print('the best result of topic-coherence'+'\t\n')
    print(params.loc[param_result['topic_coherence (u_mass)']==min(param_result['topic_coherence (u_mass)'])])
    print('\t\n')
    return params

param_result_noelse=get_model_tuning(corpus_noelse,dictionary_noelse,text_sentence_noelse,params_df)

def get_model_tuning_plot(param_result):
    for i, topic in enumerate(topic_list):
        data_iter  = param_result.loc[param_result['topic'] == topic, ['iteration',
                                                                       'topic_coherence (u_mass)']]
        plt.plot(data_iter['iteration'],data_iter['topic_coherence (u_mass)'])
        plt.title('the graph under topic {0} : iteration - topic-coherence'.format(topic))
        plt.xlabel('iteration')
        plt.ylabel('topic_coherence')
        plt.show()

    for i , iteration in enumerate(iteration_list):
        data_topic = param_result.loc[param_result['iteration'] == iteration , ['topic',
                                                                                'topic_coherence (u_mass)']]
        plt.plot(data_topic['topic'],data_topic['topic_coherence (u_mass)'])
        plt.title('the graph under iteration {0} : topic - topic-coherence'.format(iteration))
        plt.xlabel('topic')
        plt.ylabel('topic_coherence')
        plt.show()
get_model_tuning_plot(param_result_noelse)
# # 通过多次比较代码结果发现 模型在topic =32  和 34 的时 ，topic-coherence 均多次到达了最小的情况

# 进行这两组参数下，模型的结果 -- topic
candidate_params = [param_result_noelse.loc[param_result_noelse['topic_coherence (u_mass)']==\
                                      min(param_result_noelse['topic_coherence (u_mass)'])]['topic'].values[0],
                     param_result_noelse.loc[param_result_noelse['topic_coherence (u_mass)']==\
                                      min(param_result_noelse['topic_coherence (u_mass)'])]['iteration'].values[0]]
                    # [param_result_noelse[(param_result_noelse['topic']==32)].loc[param_result_noelse[(param_result_noelse['topic']==32)]['topic_coherence (u_mass)']==\
                    #                                                min(param_result_noelse[(param_result_noelse['topic']==32)]['topic_coherence (u_mass)'])]['topic'].values[0],
                    #  param_result_noelse[(param_result_noelse['topic']==32)].loc[param_result_noelse[(param_result_noelse['topic']==32)]['topic_coherence (u_mass)']==\
                    #                                                min(param_result_noelse[(param_result_noelse['topic']==32)]['topic_coherence (u_mass)'])]['iteration'].values[0]]]
best_topic = candidate_params[0]
best_iteration = candidate_params[1]

model_lda =LdaModel(corpus=corpus_noelse,num_topics=best_topic,id2word=dictionary_noelse,
                    alpha=0.01,iterations=best_iteration)
print('the topic-word distribution of topic 10 (with top-5 words) '+ '\t\n')
print(model_lda.print_topic(topicno=10,topn=5))

save_path = abs_output_path + '/lda_topic_topword.txt'
with open(save_path,'w',encoding='utf-8') as f:
    f.write('the top words in {0} topics '.format(best_topic)+ '\t\n')
    for i in range(best_topic):
        f.write('topic'+str(i)+'\r\n')
        wordid_prob = model_lda.get_topic_terms(topicid =i,topn=10)
        for wp in wordid_prob :
            word,prob = dictionary_noelse[wp[0]],wp[1]
            f.write(word+'  '+str(prob)+'\t')
        f.write('\r\n')
    f.close()


# 得到每一个文档的主题概率， get_document_topics  需要去确定的一点是 ，
labels_predict_lda = []
for i in range(len(corpus_noelse)):
    topic,prob = model_lda.get_document_topics(bow=corpus_noelse[i])[0][0],model_lda.get_document_topics(bow=corpus_noelse[i])[0][1]
    labels_predict_lda.append(topic)

# 进行结果的合并
# 将lda部分的聚类结果和mess_total=='其他'的情况进行合并

text_words_noelse['cluster_result'] = labels_predict_lda
result_noelse = text_words_noelse[['feedback_mess_total','cluster_result']]
topic_name_list=[]

text_words_else['cluster_result'] = ['{0}_其他'.format(model_lda.num_topics+1) for _ in range(len(text_words_else))]
result_else = text_words_else[['feedback_mess_total','cluster_result']]

result_final = pd.concat([result_noelse , result_else])
result_final['id'] = result_final.index.values
lda_dk_reason_henan = result_final.groupby(by='cluster_result').count()

#将最终的结果进行比较对比






# 最终的结果保存： --部分结果的数据
save_path = abs_output_path+'/lda_result_partdata.csv'
with open(save_path , 'w' ) as f :
    f.write('origin sentence' + ','+'label')
    f.write('\n')
    for index,row  in result_noelse.iterrows():
        f.write(row['feedback_mess_total']+','+str(row['cluster_result']))
        f.write('\t\n')
    f.close()

# 最终全部数据
save_path = abs_output_path+'/lda_final_result.csv'
result_final_text = result_final[['feedback_mess_total','cluster_result']]
with open(save_path , 'w' ) as f :
    f.write('cut sentence' + ','+'label')
    f.write('\n')
    for index,row in result_final_text.iterrows():
        f.write(row['feedback_mess_total']+','+ str(row['cluster_result']))
        f.write('\t\n')
    f.close()


## 对结果进行分析，最后确定每一个主题的名称,同时对于新文本进行预测
bow1 = ['每月','电费','交不上']
bow2 = ['账单','非常','看不','清楚']
new_bow =model_lda.id2word.doc2bow(bow1)
doc_topics, word_topics, phi_values = model_lda.get_document_topics(new_bow, per_word_topics=True) # per_word_topics=True 表示获取该文档在每个主题下的概率，否则只显示最大概率的情况
print('the topic 、word_topics 、phi_values of new text : {0} '.format(' '.join(bow1)))
print('topic \t\n',doc_topics,'\t\n',word_topics,'\t\n',phi_values)

all_topics = model_lda.get_document_topics(corpus_noelse, per_word_topics=True) # 获取了整个语料库的文档主题
# all_topics 的维度
cnt = 0
for doc_topics, word_topics, phi_values in all_topics[:10]:
    print('新文档:{} \n'.format(cnt),text_sentence_noelse[cnt])
    doc_topics = [(i[0],i[1]) for i in doc_topics]  #文档在每一个主题下的概率
    word_topics = [(dictionary_noelse.id2token [i[0]],i[1]) for i in word_topics]
    phi_values = [(dictionary_noelse.id2token [i[0]],i[1]) for i in phi_values ]
    print('文档主题:', doc_topics)
    print('词汇主题:', word_topics)
    print('Phi值:', phi_values)
    print(" ")
    print('-------------- \n')
    cnt+=1

## 词汇着色

color_list = ['#FFB6C1','#DC143C','#DDA0DD', '#800080','#4B0082','#6A5ACD','#4169E1' ,
              '#708090' , '#00FFFF' ,'#00CED1'  ,'#40E0D0','#7FFFAA' , '#90EE90','#FFFFE0',
              '#F0E68C','#FF7F50' ,'#FFA500','#CD5C5C','#000000','#B22222',
              '#BC8F8F','#D2691E','#FFEFD5','#FFD700','#FFFF00' ,
              '#F0FFFF','#FF6347' ,'#FFF5EE' ,'#8B4513'  ,'#8B0000', '#800000',
              '#FA8072' ,'#CD853F' ,'#FFE4C4','#FFDAB9','#0000FF']

topic_color = {}

for  i in range(best_topic):
    topic_color[i]=color_list[i]
topic_color['[]'] = '#161823'  ## 在下面进行着色的时候 如果是对于未出现的词汇则直接使用 该颜色进行代替

def color_words(model, doc, topic_colors):
    doc = model.id2word.doc2bow(doc)
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True)
    # word_topics 存放了每一个词汇的主题信息
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    word_pos = 1/len(doc)
    for word, topics in word_topics:
        if len(topics)>0 :
            ax.text(word_pos,
                    0.8,
                    model.id2word[word],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color=topic_colors[topics[0]],  # 选择可能性最大的主题
                    transform=ax.transAxes)
        else:
            ax.text(word_pos,
                    0.8,
                    model.id2word[word],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color=topic_colors['[]'],  # 选择可能性最大的主题
                    transform=ax.transAxes)
        word_pos += 0.2
    ax.set_axis_off()
    plt.show()
color_words(model_lda,bow1,topic_color)

## 上面只是对于一条文本中的词汇进行着色，下面对于整个文档中的词汇进行着色
def color_words_dict(model, dictionary,topic_color_list):
    word_topics = []
    for word_id in dictionary:
        word = str(dictionary[word_id])
        probs = model.get_term_topics(word)   # 获取每一用户的主题概率
        try:
            sorted_probs = sorted(probs , key= lambda x : x[1] ,reverse=True)
            word_topic_sorted = [ topic  for topic , pr in sorted_probs ]
            word_topics.append((word_id , word_topic_sorted))
        except IndexError:
            word_topics.append((word_id, [probs[0][0]]))
    # 设置画图的背景
    fig = plt.figure()
    ax=fig.add_axes([0,0,1,1])  ## left bottom width height 表示画板的百分比，从该画板的百分比处开始画起
    x_position = 1
    y_position = 9
    x_row_spacing = 0.2
    y_row_spacing = -0.2
    for word, topics in word_topics:
        if len(topics)>0:
            ax.text(x_position, y_position, model.id2word[word],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color=topic_color_list[topics[0]],
                    transform=ax.transAxes)
        else :
            ax.text(x_position, y_position, model.id2word[word],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color=topic_color_list['[]'],
                    transform=ax.transAxes)
        y_position -= y_row_spacing
        if  abs(y_position-0)<=0.02 :
            x_position += x_row_spacing
            y_position = 9
    ax.set_axis_off()
    plt.show()

# color_words_dict(model_lda, dictionary_noelse,topic_color) # 没有把所有的汉字都体现出来
## 引入可视化界面
# try:
#     CAN_VISUALIZE = True
#     pyLDAvis.enable_notebook()
#     from IPython.display import display
# except ImportError:
#     ValueError("SKIP: please install pyLDAvis")
#     CAN_VISUALIZE = False
# warnings.filterwarnings('ignore')
#
# prepared = pyLDAvis.gensim.prepare(model_lda, corpus_noelse, dictionary_noelse)
# pyLDAvis.show(prepared,open_browser=True)

#
