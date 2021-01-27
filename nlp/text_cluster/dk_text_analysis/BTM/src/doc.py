# -*- coding: utf-8 -*-
from .Biterm import *

class Doc():
    ws = []

    def __init__(self,s):
        self.ws = []
        self.read_doc(s)

    def read_doc(self,s):
        for w in s.split(' '):
            # print(s)
            self.ws.append(int(w))   # ws 则为切分处理后的词汇id列表
    def size(self):
        return len(self.ws)

    def get_w(self,i):
        assert(i<len(self.ws))
        return self.ws[i]

    ''' 
      Extract biterm from a document
        'win': window size for biterm extraction
        'bs': the output biterms
        通过滑动窗口的做法来不断地构造词对 ，代扣解约文本很多时候都很短，因此取win = 1
    '''
    def gen_biterms(self,bs,win=15):
        if(len(self.ws)<2):
            return
        for i in range(len(self.ws)-1):
            for j in range(i+1,min(i+win,len(self.ws))):
                bs.append(Biterm(self.ws[i],self.ws[j]))   #bs 保存的biterm函数  两个词对所对应的biterm
        return bs

