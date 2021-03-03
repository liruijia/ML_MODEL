# -*- coding: utf-8 -*-
from .pvec import *
import numpy as np
from .doc import Doc
from .sampler import *
import jieba
import re
from .Biterm import *
from zhon.hanzi import punctuation

class Model():
    bs = []
    W = 0   # vocabulary size
    K = 0   # number of topics
    n_iter = 0  # maximum number of iteration of Gibbs Sampling
    save_step = 0
    alpha = 0   # hyperparameters of p(z)
    beta = 0    # hyperparameters of p(w|z)
    nb_z = Pvec()   # n(b|z), size K*1  topic-biterm分布矩阵
    nwz = np.zeros((1,1)) # n(w,z), size K*W
    pw_b = Pvec()   # the background word distribution


    '''
        If true, the topic 0 is set to a background topic that 
        equals to the empirical word distribution. It can filter
        out common words
    '''
    has_background = False

    def __init__(self,K,W,alpha,beta,n_iter,save_step,stop_words_path,
                 dictionary_path,has_b=False):
        self.K = K
        self.W = W
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.stop_words_path = stop_words_path
        self.dictionary_path = dictionary_path
        self.save_step = save_step
        self.has_background = has_b
        self.len_biterm = None
        self.len_doc = None
        self.len_voca =None
        self.pw_b.resize(W)  #pw_b统计了词频
        self.nwz.resize((K,W))
        self.nb_z.resize(K)
        self.pz_b = None
        self.pb_d = None
        self.nb = None
        self.nz = None
        self.pw_z = None

    def run(self,doc_pt,res_dir):
        ''' doc_pt : 切分好的文本 ， res_dir
            bs 获取到的biterm(词对) 函数
        '''
        self.__loadstopwords()
        self.__loaddictionary()
        self.load_docs(doc_pt)
        self.model_init()

        print("Begin iteration")
        out_dir = res_dir + "k" + str(self.K) + "."  #保存最后的结果
        for i in range(1,self.n_iter+1):
            print("\riter "+str(i)+"/"+str(self.n_iter))
            for b in range(len(self.bs)):
                self.update_biterm(self.bs[b])   # 一次迭代 ，更新一次biterm
            if i%self.save_step == 0:
                self.save_res(out_dir)
        self.compu_pw_z()
        self.save_res(out_dir)

    def __loadstopwords(self):
        self.stop_words = []
        with open(self.stop_words_path , 'r' ,encoding='utf-8') as f:
            for word in f:
                self.stop_words.append(word)

    def __loaddictionary(self):
        self.dictionary ={}
        with open(self.dictionary_path,'r',encoding = 'utf-8') as f:
            for w in f.readlines():
                wid,word = w.strip().split(' ')[:2]
                self.dictionary[word]=wid
            f.close()
        self.len_voca = len(self.dictionary)

    def model_init(self):
        for index ,biterm in enumerate(self.bs):
            k = uni_sample(self.K)
            self.assign_biterm_topic(biterm,k)
        self.total_ds = {}
        for  index, bitermb in enumerate(self.bs):
            if bitermb not in self.total_ds.keys():
                self.total_ds[bitermb]=index
        # self.len_biterm = len(self.total_ds)
        # self.pz_b = np.zeros((self.len_biterm ,self.K))
        # self.pb_d = np.zeros((self.len_doc , self.len_biterm))
        # self.nd_b = np.zeros((self.len_doc ,self.len_biterm))
        # self.nz = np.zeros((self.K))

    def load_docs(self,docs_pt):
        '''
          docs_pt : 已经是经过id2word转换后的文档  每一则文本均用 词汇的id 来表示
        '''
        print("load docs: " + docs_pt)
        rf = open(docs_pt,encoding='utf-8')
        if not rf:
            print("file not found: " + docs_pt)
        self.len_doc =0
        self.doc_ds = []
        for line in rf.readlines():
            d = Doc(line)
            self.len_doc += 1
            biterms = []
            d.gen_biterms(biterms)
            # statistic the empirical word distribution d.size 即指获取到的biterm词对数量
            for i in range(d.size()):
                w = d.get_w(i)
                self.pw_b[w] += 1
            for b in biterms:
                self.bs.append(b)
            self.doc_ds.append(biterms)
        self.pw_b.normalize()

        # print('********保存bs******')
        # print('the length of ds ' , len(self.bs))
        # save_path = '***'
        # with open(save_path, 'w',encoding = 'utf-8') as f:
        #     for b in self.bs:
        #         f.write(b)
        #         f.write('\n')
        #     f.close()

    def update_biterm(self,bi):
        self.reset_biterm_topic(bi)  # 根据词对进行做类似于doc-topic , topic-word的更新

        # comput p(z|b)
        pz = Pvec()
        self.comput_pz_b(bi,pz)

        # sample topic for biterm b
        k = mul_sample(pz.to_vector())
        self.assign_biterm_topic(bi,k)

    def reset_biterm_topic(self,bi ):
        k = bi.get_z()
        w1 = bi.get_wi()
        w2 = bi.get_wj()
        bid = self.total_ds[bi]

        self.nb_z[k] -= 1
        self.nwz[k][w1] -= 1
        self.nwz[k][w2] -= 1
        # self.nz[k] -=1
        # self.nd_b[doc_id ,bid]-=1
        assert(self.nb_z[k] > -10e-7 and self.nwz[k][w1] > -10e-7 and self.nwz[k][w2] > -10e-7)
        bi.reset_z() # 重新赋予新的主题

    def assign_biterm_topic(self,bi,k): # 给biterm-bi 产生的新的主题
        bi.set_z(k)
        w1 = bi.get_wi()
        w2 = bi.get_wj()
        # bid = self.total_ds[bi]
        self.nb_z[k] += 1
        self.nwz[k][w1] += 1
        self.nwz[k][w2] += 1


    def comput_pz_b(self,bi,pz):  # 计算 p(z|b)
        pz.resize(self.K)
        w1 = bi.get_wi()
        w2 = bi.get_wj()

        for k in range(self.K):
            if (self.has_background and k == 0) :
                pw1k = self.pw_b[w1]
                pw2k = self.pw_b[w2]
            else:
                pw1k = (self.nwz[k][w1] + self.beta) / (2 * self.nb_z[k] + self.W * self.beta)
                pw2k = (self.nwz[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.W * self.beta)

            pk = (self.nb_z[k] + self.alpha) / (len(self.bs) + self.K * self.alpha)
            pz[k] = pk * pw1k * pw2k

    def compu_pw_z(self):
        '''计算p(w|z) = \frac{nw_z + self.beta}{n_z[k]+self.W*self.beta}
        nwz 即 nw_z
        '''
        pw_z = np.zeros((self.K,self.W))
        for k in range(self.K):
            for w in range(self.W):
                pw_z[k][w] = (self.nwz[k][w] + self.beta) / (self.nb_z[k] * 2 + self.W * self.beta)
        self.pw_z = pw_z

    def build_Biterms(self, sentence):
        win = 1  # 设置窗口大小
        biterms = []
        if not sentence or len(sentence) <= 1:
            return biterms
        for i in range(len(sentence) - 1):
            for j in range(i + 1, min(i + win + 1, len(sentence))):
                biterms.append(Biterm(int(sentence[i]), int(sentence[j])))
        return biterms


    def __textProcessing(self,sentence):
        new_se=re.sub(r'[%s,\t,\\]+'%punctuation,' ',sentence)
        new_s = jieba.cut(new_se)
        new = []
        for word in new_s:
            if word not in self.stop_words :
                #print('filter stop_words',word)
                new.append(word)
        return new

    def __SentenceProcess(self, sentence,sentence_type='wordid_list'):
        if sentence_type == 'text':
            words = self.__textProcessing(sentence)
            words_id = []
            for w in words:
            #print('in wordid ' ,w)
                words_id.append(self.dictionary[w])
        elif sentence_type == 'wordid_list':
            words_id = sentence
        return self.build_Biterms(words_id)

    def sentence_topic(self, sentence, sentence_type = 'wordid_list',topic_num=1, min_pro=0.02):
        # p(z|d)= p(z|w)p(w|d)
        words_id = self.__SentenceProcess(sentence,sentence_type)
        print(words_id,'***')
        topic_pro = [0.0] * self.K
        sentence_word_dic = [0] * self.len_voca
        print('the length of sentence' , len(words_id))
        weigth = 1.0 / len(words_id)
        for i in range(len(words_id)):
            word_id1 = words_id[i].get_wi()
            word_id2 = words_id[i].get_wj()
            sentence_word_dic[word_id1] = weigth
            sentence_word_dic[word_id2] = weigth
        for i in range(self.K):
            topic_pro[i] = sum(map(lambda x, y: x * y, self.nwz[i], sentence_word_dic))
            # print('topic_pro[i]',topic_pro[i],sentence_word_dic ,self.nwz[i])
        sum_pro = sum(topic_pro)
        print('sum_pro',sum_pro)
        topic_pro = map(lambda x: x / sum_pro, topic_pro)
        # print topic_pro
        min_result = list(zip(topic_pro, range(self.K)))
        min_result = sorted(min_result, key=lambda x: x[0], reverse=True)
        result = []
        print(min_result)
        for re in min_result:
            if re[0] > min_pro:
                result.append(re)
        # result 返回了该sentence所对应的主题以及主题下的概率
        return result[:topic_num]

    def infer_sentence_topics(self, sentence, sentence_type='wordid_list',topic_num=1):
        # p(z|d) = p(z|b)p(b|d)
        sentence_biterms = self.__SentenceProcess(sentence,sentence_type)
        topic_pro = [0] * self.K
        # 短文本分析中，p (b|d) = nd_b/doc(nd_b)  doc(nd_b) 表示 计算的query 的所有biterm的计数
        bit_size = len(sentence_biterms)
        #print('this sentence has {0} biterms-words pair'.format(bit_size))
        if not sentence_biterms:
            return None
        for bit in sentence_biterms:
            pz = [0] * self.K
            opz = Pvec(pvec_v=pz)
            self.comput_pz_b(bit, opz)
            pz_sum = sum(opz.p)
            pz = map(lambda pzk: pzk / pz_sum, opz.p)
            for x, y in zip(range(self.K), pz):
                topic_pro[x] += y / bit_size
        min_result = list(zip(topic_pro, range(self.K)))
        min_result = sorted(min_result, key=lambda x: x[0], reverse=True)[:topic_num] # topic_pro topic
        return min_result

    def save_res(self,res_dir):
        # 保存计算得到的主题概率 以及 主题-词分布
        pt = res_dir + "pz"
        print("\nwrite p(z): "+pt)
        self.save_pz(pt)

        pt2 = res_dir + "pw_z"
        print("write p(w|z): "+pt2)
        self.save_pw_z(pt2)

    # p(z) is determinated by the overall proportions of biterms in it
    def save_pz(self,pt):
        pz = Pvec(pvec_v=self.nb_z)
        pz.normalize(self.alpha)
        pz.write(pt)

    def save_pw_z(self,pt):
        pw_z = np.ones((self.K,self.W))
        wf = open(pt,'w')
        for k in range(self.K):
            for w in range(self.W):
                pw_z[k][w] = (self.nwz[k][w] + self.beta) / (self.nb_z[k] * 2 + self.W * self.beta)
                wf.write(str(pw_z[k][w]) + ' ')
            wf.write("\n")
