from kmodes.kprototypes import  KPrototypes
import numpy as np
from sklearn.metrics import calinski_harabasz_score ,silhouette_score
from  kmodes.util import dissim
num_dissim = dissim.euclidean_dissim
cat_dissim = dissim.matching_dissim
def run(data,K,save_path,loss_type = 'calinski_harabasz_score'):
    '''
    :param data:
    :param K:
    :param loss_type : 损失函数的类型，只有两种 calinski_harabasz_score 以及silhouette_score ,均是值越大越好
    :return:
    '''
    global  num_dissim
    global  cat_dissim
    save_pt = save_path + '/kprototypes_reco_result.csv'
    ch=[]
    sl=[]
    iter_list = [50,100,150,200]
    for iter in iter_list :
        model = KPrototypes(n_clusters=K,max_iter=iter,num_dissim= num_dissim, cat_dissim=cat_dissim )
        predict = model.fit_predict(data,categorical=[1,2])
        if loss_type== 'calinski_harabasz_score':
            ch.append(calinski_harabasz_score(data,predict))
        else :
            sl.append(silhouette_score(data,predict))
    if loss_type == 'calinski_harabasz_score':
        best_iter_id = np.argmax(ch)
        print('the best score of calinski_harabasz_score : {0}'.format(ch[best_iter_id]))
        best_iter = iter_list[best_iter_id]
    else:
        best_iter_id =np.argmax(sl)
        best_iter =iter_list[best_iter_id]
    # print('******')
    # print(model.cluster_centroids_)
    # print(len(model.cluster_centroids_) ,len(model.cluster_centroids_[0]))
    # 开始最终的训练，使用得到的iter
    print('the best  iter' ,best_iter)
    model = KPrototypes(n_clusters=K,max_iter=best_iter,num_dissim= num_dissim, cat_dissim=cat_dissim )
    predict = model.fit_predict(data,categorical=[1,2])
    centroids = model.cluster_centroids_
    print('\t\r the centroids is ')
    return predict,centroids


if __name__ == '__main__' :
    data = np.random.choice(20,(100,10))
    save_path = './calinski_harabasz_score.csv'
    predict = run(data,3 ,save_path,loss_type = 'calinski_harabasz_score')









