import numpy as np
from numpy import genfromtxt
from logisticRegression_cv import lr
from run_fasttext import fasttext

train_source =  np.genfromtxt(fasttext('train.txt'))
train_source_labels = np.genfromtxt('train_labels.txt')
validation_source =  np.genfromtxt(fasttext('validation.txt'))
validation_source_labels = np.genfromtxt('validation_labels.txt')

# data_source = []
# for x in train_source:
    # data_source.append(x)
# for x in validation_source:
    # data_source.append(x)

# data_source_labels = []
# for y in train_source_labels:
    # data_source_labels.append(y)
# for y in validation_source_labels:
    # data_source_labels.append(y)
    
scores = lr(train_source, train_source_labels, validation_source, validation_source_labels)
print(scores)
