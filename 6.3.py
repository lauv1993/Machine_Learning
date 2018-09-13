# coding:utf8
import numpy as np
import pandas as pd
from sklearn import svm
import random

data = pd.read_csv(filepath_or_buffer='iris.csv')

columns = data.columns
attributes = columns[:-1]
label = columns[-1]

sample_frame = data[attributes]
label_series = data[label]

samples = sample_frame.values
labels = label_series.tolist()

#选取前n个作为训练数据
n = 40
train_X = np.vstack((samples[:n], samples[50:50+n]))
train_Y = np.hstack((labels[:n], labels[50:50+n]))

test_X = np.vstack((samples[n:50], samples[50+n:]))
test_Y = np.hstack((labels[n:50], labels[50+n:]))

clf = svm.SVC(kernel="linear")
clf.fit(train_X, train_Y)
print clf.support_vectors_
print clf.support_
print clf.n_support_

#verifing linear kernel
print clf.predict(test_X)

clf2 = svm.SVC(kernel="rbf")
clf2.fit(train_X, train_Y)
print clf2.support_vectors_
print clf2.support_
print clf2.n_support_

#verifing linear kernel
print clf2.predict(test_X)




