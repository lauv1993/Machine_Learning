# coding:utf8
import numpy as np
import pandas as pd
from sklearn import svm

data = pd.read_csv(filepath_or_buffer = 'watermelon3.0alpha.csv')

columns = data.columns
attributes = columns[-3:-1]
label = columns[-1]
samples_frame = data[attributes]
label_series = data[label]
samples = samples_frame.values
labels = label_series.tolist()


#linear Kernal
clf = svm.SVC(kernel='linear')
clf.fit(samples, labels)
#get support vectors
print clf.support_vectors_
#get indices of support vectors
print clf.support_
#get number of support vectors for each classes
print clf.n_support_

#gaussian kernel
clf2 = svm.SVC(kernel='rbf')
clf2.fit(samples, labels)
#支持向量
print clf2.support_vectors_
#支持向量的索引
print clf2.support_
#各个类的数量
print clf2.n_support_