"""
File: logistic_regression.py
Description: Train logistic regression classifier in transfer learning setting
"""

import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import statistics

def lr(train_source, train_source_labels, validation_source, validation_source_labels):

    data_source = []
    for x in train_source:
        data_source.append(x)
    for x in validation_source:
        data_source.append(x)

    data_source_labels = []
    for y in train_source_labels:
        data_source_labels.append(y)
    for y in validation_source_labels:
        data_source_labels.append(y)
    lr = linear_model.LogisticRegression(C=1e5)
    #clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X, y)
    scores = cross_val_score(lr, data_source, data_source_labels, cv=5, scoring='f1_weighted')
    return  statistics.mean(scores)
