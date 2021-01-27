# 利用BTM 模型进行文档的分类
# 主要是获取到document - topic 分布即可进行相应的分类了
#

def display_keyword_of_document(corpus_text,predict_label,dictionary,pw_z_dir,topic_num,docn=20,topn=5):
    pw_z = load_pw_z(pw_z_dir,topic_num)
    for i in range(docn):
        print('**** the sentence  **** \r\n')
        print(corpus_text[i])
        print('**** the response topic **** \r\n')
        print(predict_label[i])
        print('**** the response keywords **** \r\n')
        word_id = range(len(pw_z[predict_label[i]]))
        word_prob = list(zip(word_id,pw_z[predict_label[i]]))  # zip(wordid ,  wordprob)
        prob_result = sorted(word_prob, key=lambda x :x[1],reverse=True)[:topn]
        #print(prob_result)
        for wordpro in prob_result :
            wid , _ = wordpro[0],wordpro[1]
            print(dictionary[str(wid)]+'\t')

def load_pw_z(pw_z_dir,topic_num):
    pw_z=[]
    with open(pw_z_dir,'r',encoding='utf-8') as f:
        for line in f.readlines():
            lines=list(map(float,line.strip().split(' ')))
            pw_z.append(lines)
    return pw_z

def load_document(dwid_pt,text_dir):
    corpus_list = []
    with open(dwid_pt ,'r',encoding='utf-8') as f :
        for line in f.readlines():
            s = line.strip().split(' ')
            corpus_list.append(s)
    text_corpus = []
    with open(text_dir,'r',encoding = 'utf-8') as f :
        for line in f.readlines():
            s=line.strip()
            text_corpus.append(s)
    return corpus_list,text_corpus

def load_voca(voca_dir):
    dictionary={}
    with open(voca_dir , 'r',encoding ='utf-8') as f:
        for line in f.readlines():
            wid , word =line.strip().split(' ')[:2]
            if wid not in dictionary.keys():
                dictionary[wid]=word
        f.close()
    return dictionary

def save_classify_result(predict_label,text_corpus,save_path):
    with open(save_path , 'w' ,encoding='utf-8') as f:
        f.write('sentence' + ',' + 'btm_label')
        f.write('\r')
        for i in range(len(predict_label)):
            f.write(text_corpus[i]+','+str(predict_label[i])+'\r')
        f.close()

def run_document_classify(model, dwid_pt , text_dir , voca_dir,pw_z_dir, save_path):
    corpus_list,text_corpus = load_document(dwid_pt,text_dir)
    predict = []
    for i in range(len(corpus_list)):
        label = model.infer_sentence_topics(corpus_list[i],sentence_type='wordid_list')
        #print(label)
        if label is None :
            predict.append(None)
        else:
            predict.append(label[0][1])
    save_classify_result(predict,text_corpus,save_path)
    topic_num=model.K
    dictionary = load_voca(voca_dir)
    display_keyword_of_document(text_corpus,predict,dictionary,pw_z_dir,topic_num,docn=20,topn=5)