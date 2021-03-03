# 对于tf-idf的关键字信息进行分析

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

abs_path = 'activity_relate/dk_henan_pay/use_python_analysis/output_data/'
key_words_path = abs_path + 'keyword_result.csv'
key_word_df = pd.read_csv(key_words_path,engine='python')
key_words = key_word_df['keyword'].values.tolist()
senten= ' '.join(map(str,key_words))

model=WordCloud(background_color='white',min_font_size=5,font_path='msyh.ttc',max_words=3000
                ,max_font_size=1000)
model.generate(text=senten )
plt.figure("词云图")
plt.imshow(model)
plt.axis("off")
plt.show()

# kmeans结果分析，得到聚类结果之后，对于同一类别下的文本进行汇合

kmeans_path=abs_path + 'kmeans_result.csv'
kmeans_df = pd.read_csv(kmeans_path , engine = 'python')
cluster_num = len(kmeans_df['label_kmeans	'].unique())
result_kmeans ={}
for i in range(cluster_num):
    op = kmeans_df.loc[kmeans_df['label_kmeans']==i,'sentence'].values.tolist()
    result_kmeans[i]=np.unique(op).tolist()

with open(abs_path + 'kmeans_analysis.csv' , 'w',encoding='utf-8') as f:
    for key,op in result_kmeans.items():
        f.write('label:'+str(key)+'\r\n')
        # print(op)
        f.write('sentence: '+' '.join(op)+'\r\n')
    f.close()

