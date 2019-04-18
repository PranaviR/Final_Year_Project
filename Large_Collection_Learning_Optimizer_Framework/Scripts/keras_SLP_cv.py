import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy
import statistics


def SLP(trainFeatures, trainLabels, validationFeatures, validationLabels):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    x_train = np.array(trainFeatures)
    y_train = np.array(trainLabels)
    x_test = np.array(validationFeatures)
    y_test = np.array(validationLabels)
    

    data_source = []
    for x in x_train:
        data_source.append(x)
    for x in x_test:
        data_source.append(x)

    data_source_labels = []
    for y in y_train:
        data_source_labels.append(y)
    for y in y_test:
        data_source_labels.append(y)
    
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    
    for train_index,test_index in kfold.split(data_source, data_source_labels):
    #create model
        model = Sequential()
        #model.add(Dense(units=64, activation='sigmoid', input_dim=768))
        model.add(Dense(units=2, activation='softmax', input_dim=300))
        #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='sgd', metrics=['accuracy'])
        #train_x, test_x = data_source[train_index], data_source[test_index]
        #train_y, test_y = data_source_labels[train_index], data_source_labels[test_index]
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for index in train_index:
            train_x.append(data_source[index])
            train_y.append(data_source_labels[index])
        for index in test_index:
            test_x.append(data_source[index])
            test_y.append(data_source_labels[index])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        model.fit(train_x, train_y, epochs=150, batch_size=10) 
        predictions = model.predict_classes(test_x)
        f1 = f1_score(test_y, predictions, average = 'weighted')
        cvscores.append(f1)
    return statistics.mean(cvscores)
