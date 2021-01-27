#!/usr/bin/env python

# Function: translate the results from BTM
# Input:
#    mat/pw_z.k20

import sys
print(sys.getdefaultencoding())
# return:    {wid:w, ...}
def read_voca(pt):
    voca = {}
    #print('pt',pt)
    for l in open(pt,encoding='utf-8',errors='ignore').readlines():
        #print(l)
        wid, w = l.strip().split(' ')
        voca[int(wid)] = w
    return voca

def read_pz(pt):
    return [float(p) for p in open(pt).readline().split()]

def dispTopics(pt, voca, pz,save_path,topn=10):
    save_pt = save_path + '/topic_words.txt'
    k = 0
    topics = []
    pw_z = []
    for l in open(pt):
        vs = [float(v) for v in l.split()]
        pw_z.append(vs)
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d:d[1], reverse=True)
        tmps = ' '.join(['%s:%f' % (voca[w],v) for w,v in wvs[:10]])
        topics.append((pz[k], tmps))
        k += 1
    topic_num = len(pw_z)
    voca_num = len(pw_z[0])
    with open(save_pt , 'w' ,encoding = 'utf-8') as f:
        for i in range(topic_num):
            f.write('topic:'+str(i))
            lk = list(zip(range(voca_num),pw_z[i]))
            sote = sorted(lk , key=lambda x : x[1] ,reverse= True)[:topn]
            f.write('\r\n words: \t\t')
            for word_p in sote:
                word,p=voca[word_p[0]],word_p[1]
                f.write(word+'\t')
            f.write('\r\n')
        f.close()





def run_topicDicplay(argv):
    if len(argv) < 4:
        print('Usage: python %s <model_dir> <K> <voca_pt>' % argv[0])
        print('\tmodel_dir    the output dir of BTM')
        print('\tK    the number of topics')
        print('\tvoca_pt    the vocabulary file')
        print('\tsave_path   the save path of top_words in every words ')
        exit(1)
    model_dir = argv[1]
    K = int(argv[2])
    voca_pt = argv[3]
    save_path = argv[4]
    voca = read_voca(voca_pt)    
    W = len(voca)
    print('K:%d, n(W):%d' % (K, W))
    pz_pt = model_dir + 'k%d.pz' % K
    pz = read_pz(pz_pt)
    zw_pt = model_dir + 'k%d.pw_z' %  K
    dispTopics(zw_pt, voca, pz,save_path)

