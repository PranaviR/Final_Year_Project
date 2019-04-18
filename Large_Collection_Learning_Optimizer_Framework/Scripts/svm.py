"""
File: logistic_regression.py
Description: Train logistic regression classifier in transfer learning setting
"""

import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def svm(train_source, train_source_labels, test_source, test_source_labels):

	# Define training and test splits
	# train_source = np.genfromtxt(source_file)
	# test_source = np.genfromtxt(test_file)

	# train_source_labels = np.genfromtxt(source_file_labels)
	# test_source_labels = np.genfromtxt(test_file_labels)

	clf = SVC(gamma = 'scale')
	clf.fit(train_source, train_source_labels)

	predictions = clf.predict(test_source)
	return  f1_score(test_source_labels, predictions)
