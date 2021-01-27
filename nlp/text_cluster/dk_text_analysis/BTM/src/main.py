# -*- coding: utf-8 -*-
import time
from . import Model
from .indexDocs import run_indexDocs
from .estimator import *
from .topicDisplay import run_topicDicplay
from .document_classify import run_document_classify
from . import topicModel_umass


def usage() :
    print("Training Usage: \
    btm est <K> <W> <alpha> <beta> <n_iter> <save_step> <docs_pt> <model_dir>\n\
    \tK  int, number of topics, like 20\n \
    \tW  int, size of vocabulary\n \
    \talpha   double, Pymmetric Dirichlet prior of P(z), like 1.0\n \
    \tbeta    double, Pymmetric Dirichlet prior of P(w|z), like 0.01\n \
    \tn_iter  int, number of iterations of Gibbs sampling\n \
    \tsave_step   int, steps to save the results\n \
    \tdocs_pt     string, path of training docs\n \
    \tmodel_dir   string, output directory")


def BTM(argvs):
    if(len(argvs)<4):
        usage()
    else:
        if (argvs[0] == "est"):
            K = argvs[1]
            W = argvs[2]
            alpha = argvs[3]
            beta = argvs[4]
            n_iter = argvs[5]
            save_step = argvs[6]
            docs_pt = argvs[7]
            dir = argvs[8]
            stop_words_path =argvs[9]
            dictionary_path = argvs[10]
            print("Run BTM, K="+str(K)+", W="+str(W)+", alpha="+str(alpha)+", beta="+str(beta)+", n_iter="+str(n_iter)+", save_step="+str(save_step)+\
                  ", stop_words_path="+stop_words_path+", dictionary_path="+dictionary_path+"=====")
            clock_start = time.time()
            model = Model(K, W, alpha, beta, n_iter, save_step,stop_words_path,dictionary_path)
            model.run(docs_pt,dir)
            sentence_new='更换住处'
            result=model.sentence_topic(sentence=sentence_new,sentence_type='text')
            print('****  print result  **** \t\n')
            print(result)
            result0 = model.infer_sentence_topics(sentence=sentence_new,sentence_type = 'text') # 只求sentence-topic
            print('\t\n**** infer sentence result ******\t\n')
            print(result0)
            clock_end = time.time()
            print("procedure time : "+str(clock_end-clock_start))
            return model
        else:
            usage()

def Test(argvs1):
    doc_pt= argvs1[0]
    model_dir = argvs1[1]
    topic_list = argvs1[2]
    iteration_list = argvs1[3]
    argvs=argvs1[4:]
    model = estimator(doc_pt,model_dir,topic_list , iteration_list,argvs)
    model.estimator(plot_result=False)

if __name__ ==  "__main__":
    mode = "est"
    K = 25   # 主题个数
    W = None
    alpha = K/50
    beta = 0.01
    n_iter = 300
    save_step = 100
    abs_path='C:/Users/bdruijiali/Desktop/团队/data/activity_relate/BTM'
    dir = abs_path+"/output/"
    input_dir = abs_path+"/input/"
    model_dir = dir + "model/"
    voca_pt = dir + "voca.txt"
    dwid_pt = dir + "doc_wids.txt"
    doc_pt = input_dir + "dk_input.txt"
    stop_dir = input_dir + 'stop_words.txt'
    pred_dir = dir + 'predict_btm.csv'
    print("=============== Index Docs =============")
    W = run_indexDocs(['indexDocs',doc_pt,dwid_pt,voca_pt])

    print("W : "+str(W))

    argvs = []
    argvs.append(mode)
    argvs.append(K)
    argvs.append(W)
    argvs.append(alpha)
    argvs.append(beta)
    argvs.append(n_iter)
    argvs.append(save_step)
    argvs.append(dwid_pt)
    argvs.append(model_dir)
    argvs.append(stop_dir)
    argvs.append(voca_pt)


    print("=============== Topic Learning =============")
    model=BTM(argvs)


    print("================ Topic Display =============")
    run_topicDicplay(['topicDisplay',model_dir,K,voca_pt,dir])





    print("=============== Get document Classify ===========")
    pw_z_dir = model_dir + 'k{0}.pw_z'.format(model.K)
    run_document_classify(model, dwid_pt , doc_pt , voca_pt,pw_z_dir, pred_dir)

    # print('============= estimator ===================  ')
    # topic_list = [15,18,20,23,25,28,30]
    # iteration_list =[50,100,200,300,350]
    # argvs1 = []
    # argvs1.append(dwid_pt)
    # argvs1.append(model_dir)
    # argvs1.append(topic_list)
    # argvs1.append(iteration_list)
    # argvs1.append(W)
    # argvs1.append(alpha)
    # argvs1.append(beta)
    # argvs1.append(save_step)
    # argvs1.append(stop_dir)
    # argvs1.append(voca_pt)
    # Test(argvs1)