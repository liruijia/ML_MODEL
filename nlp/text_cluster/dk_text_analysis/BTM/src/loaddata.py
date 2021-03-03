#coding : utf-8


import pandas  as pd
import jieba

from pylab import mpl

from gensim import corpora
import re
from zhon.hanzi import punctuation
jieba.load_userdict()
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

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



if __name__ == '__main__':
    abs_input_path='**'
    abs_output_path = '**'
    jieba.load_userdict(abs_input_path+'/代扣词典.txt')
    path = abs_input_path+'/train.csv'
    train = pd.read_csv(path,engine='python')
    map_x = {1:'缴费失败',2:'担心资金安全',3:'更换住处',4:'其他',5:'手误签约',6:'缴费重复'}
    train['feedback_mess_total']=train['feedback_msg']
    train.loc[train['feedback_msg']=='missing','feedback_mess_total']=train.loc[train['feedback_msg']=="missing",
                                                                                'feedback_code'].map(map_x)

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

    stop_path =abs_input_path+'/stop_words.txt'
    stop_words = []
    with open(stop_path , 'r', encoding = 'utf-8') as f:
        for line in f:
            stop_words.append(line)
        f.close()
    stop_words.append(['元','说','了','多元','的','从','啊','10','30','100','300','扣','度','我','换了','地区'
                          ,'块','块钱','佳木斯市','前进区','会','时候','后','由','要','交','号','想','等','赤峰',''])
    cut_text=[]
    vocabulary_list = []

    text_sentence_noelse,cut_text_noelse,vocabulary_list_noelse,dictionary_noelse,corpus_noelse = get_cut_text(text_words_noelse0.values)
    save_cuttext_path = abs_output_path + '/dk_input.txt'
    with open(save_cuttext_path , 'w' ,encoding = 'utf-8') as f :
        for index, sentence in enumerate(text_sentence_noelse):
            f.write(' '.join(sentence))
            f.write('\t\n')
        f.close()
