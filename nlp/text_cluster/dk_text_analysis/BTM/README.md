# BTMpy
BTM in python

## what is BTM

It's a model proposed by Xiaohui Yan

paper : A Biterm model for short texts

i use it to analysis 'danmu'

## how to use

  every parameter are set in src/main.py
  go in src/main.py
  run it

  or just

  python src/main.py

关于数据

    input:
      doc_pt  : 切分好的短文本
    output:
      voca_pt : id2word词典路径
      doc_wids : doc-worid 格式的

关于BTM代码：
    
    原Git下的代码，需要修改：voc.bat这一部分、win参数的设置，以及一些乱起八糟格式、语句的修改
    
    在原来的基础上的添加求解pz_d的部分  用于求解文档的分类 (这一部分又涉及到一些乱七八糟的内容！！！) 包括新增： 对于新sentence数据处理的部分，求解sentence-topic
    
    toicDisplay.py 最后的结果展示
    
    indexDocs.py 做文本格式的转换 类似于corpora id2word 
    
    Model.py  BTM模型  在此基础上添加doc-topic求解的部分 （未修改完）
    
    Biterm.py 获取biterm词对中的值
    
    doc.py  滑动窗口组合biterm 
    
    sampler 随机函数 
    新增： 
         loaddata  加载代扣数据集
         main.py 中 sentence-topic 求解 ，新sentence处理部分
         documen_classify 是准备利用得到的doc-topic概率进行文档分类  
         tm_umass(基本写完，但是还没测试 不知道对不对啊希望不要被打脸  --- perplexity，u_mass) 
         estimator.py

关于结果： 
    需要利用doc-topic 进行文档分类
    
    关键词展示 

通过最后的使用，感觉文本切分那一块还得处理一下，总感觉切分的效果没有想象的好，得进行优化！！！！
