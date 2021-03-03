
# train_model.py  & henan_dk.py 

包含：数据处理模块和模型训练两个部分

   * 数据处理模块：获取各个分析阶段使用到的数据格式

     数据导入： 对于数据进行简单的数据处理（数据填充）、文本分词 、 去停用词
     
     适用于word2vec : 将导入的数据进行词向量求取，并最后整理成为可以应用于kmeans的格式
     
     适用于lda模型

     参数说明：   
        
         data_train : 导入进来的全部的数据集
         
         data_text : 处理好的文本数据（切分、去符号、去停用词） 格式： [['一则切分好的样本']]
            例： [['明天 天气 真好']]
         
         data_text_sentence : 使用的word2vec模型的样本 ，格式：[[''，'','']] 将上述的样例按照分隔符进行划分
            例： [['明天','天气','真好']]
         
         data_matrix : 使用于kmeans的数据 
         
        
 
   * 模型训练模块：主要包括使用不同的模型的进行训练 （目前只有 Kmeans 和 LDA ）
   
   * 方法：
   
        tfidf + kmeans + word2vec :
        
            利用tfidf得到的tfidf矩阵进行聚类分析
            利用tfidf进行关键字获取 -- 更加倾向于高频词汇
            利用word2vec方法得到词向量 & 文本向量 进行聚类分析
        
        kmeans : (部分指标是需要真实标签，部分指标则不需要)
        
            kmeans算法使用到的矩阵：
                tf -idf 
                word2vec
         
            metrics.adjusted_mutual_info_score(…[, …])	
            metrics.adjusted_rand_score(labels_true, …)	
            metrics.calinski_harabasz_score(X, labels)	CH分数（值越大越好） --  簇间方差/簇内方差 
            metrics.davies_bouldin_score(X, labels)
            metrics.completeness_score(labels_true, …)	
            metrics.cluster.contingency_matrix(…[, …])	
            metrics.fowlkes_mallows_score(labels_true, …)	
            metrics.homogeneity_completeness_v_measure(…)	
            metrics.homogeneity_score(labels_true, …)	
            metrics.mutual_info_score(labels_true, …)	
            metrics.normalized_mutual_info_score(…[, …])	
            metrics.silhouette_score(X, labels[, …]) 轮廓系数(值越大越好)  --利用样本的簇内不相似度和簇间不相似度定义	， 具有高度聚类 1  不正确聚类 -1  重叠接近0
            metrics.silhouette_samples(X, labels[, metric])
            metrics.v_measure_score(labels_true, labels_pred)
         
        lda:
            
            lda不常用于短文本
            topic-coherence  主题一致性
            preplexity       困惑度
            
            lda --分类
            lda --关键字提取
        
        btm (Biterm Topic Model):
                
       
        twe:
            
               lda模型对于短文本的效果不是很好，无法抓取低频词汇 
               twe模型= lda + word2vec 
               将词向量和主题向量结合起来进行考虑
         
        
   * 最终的结果: 
      kmeans :   kmeans_result.csv 
       
         cluster 0 : 担心资金安全
         cluster 1 : 自动扣款额度问题
         cluster 2 : 缴费失败
         cluster 3 :  扣费金额问题
         cluster 4 ：扣费顺序 + 扣款重复 + 扣款失败
         cluster 5 ：未扣款成功
         cluster 8 ： 有余额仍扣款
         cluster 10 ： 手误签约
         ......
         有的类别有交叉因素 
      
       lda : 
