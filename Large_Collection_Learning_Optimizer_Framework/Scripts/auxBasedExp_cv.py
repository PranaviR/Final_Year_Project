import random
import numpy as np
from numpy import genfromtxt
from logisticRegression_cv import lr
from run_fasttext import fasttext
from keras_SLP_cv import SLP
#from run_process_vec import process

from cosineSim import cosineSimilarity

#Logging configuration for logging data
from datetime import datetime

import logging

now = datetime.now().strftime('%Y-%m-%d-%H-%M')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

fh = logging.FileHandler("{}.log" .format(now))
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger = logging.getLogger('auxBasedExp')
logger.setLevel(logging.DEBUG)

logger.addHandler(fh)
logger.addHandler(ch)

logger.info("Trial4 Hurricane Matthew cv")
logger.info("\n")

#pre_trained_fasttext_model = "cc.en.bin.300"

trainingFile = 'train.txt'
trainLabelsFile = 'train_labels.txt'

validationFile = 'validation.txt'
validationLabelsFile = 'validation_labels.txt'

# Run run_fasttext.py to create feature vectors 
# Run run_process_vec.py to add commas to feature vectors

trainingResultFile = fasttext(trainingFile)
trainFeatures = np.genfromtxt(trainingResultFile)
trainLabels = np.genfromtxt(trainLabelsFile)

validationResultFile = fasttext(validationFile)
validationFeatures = np.genfromtxt(validationResultFile)
validationLabels = np.genfromtxt(validationLabelsFile)


#auxiliaryDataFile = fasttext('auxiliary.txt')
#auxList = np.genfromtxt(auxiliaryDataFile)

with open('auxiliary.txt') as f:
    auxTweets = f.read().splitlines()

######################################################################################################
#JUNK
# trainingResultFile = process(fasttext(trainingFile))
# validationResultFile = process(fasttext(validationFile))
# auxiliaryDataFile = process(fasttext('auxiliary.txt'))
#figure out how to remove the last empty line
#trainingFeatures = ''
#validationFeatures = ''
######################################################################################################

positiveTrainingFeatures = [y for x, y in zip(trainLabels, trainFeatures) if x == 1.0]
negativeTrainingFeatures = [y for x, y in zip(trainLabels, trainFeatures) if x == 0.0]

######################################################################################################
##parse through featureVectorFile and Labels file and make it into a single file

# with open('source.txt', 'w') as file3:
    # with open(trainingFeatures, 'r') as file1:
        # with open(trainLabelsFile, 'r') as file2:
            # for line1, line2 in zip(file1, file2):
                # file3.write(line1 + ',' + line2)
				
# file1.close()
# file2.close()
# file3.close()
				
# with open('source_Validation.txt', 'w') as file3:
    # with open(validationFeatures, 'r') as file1:
        # with open(validationLabelsFile, 'r') as file2:
            # for line1, line2 in zip(file1, file2):
                # file3.write(line1 + ',' + line2)
# file1.close()
# file2.close()
# file3.close()

# trainingDataFile = 'source.txt'
# validationDataFile = 'source_Validation.txt'
######################################################################################################

			
minSimilarityThreshold = 0.65
similarityWindowSize = 0.05

numberOfIterations = 0
bestIteration = 0

maxNoOfIterations = 20
thresholdF1 = 0.98
auxThresholdexpectation = 0.01

initial_F1 = SLP(trainFeatures, trainLabels, validationFeatures, validationLabels) 
print(initial_F1)
f1 = initial_F1
logger.info(str(f1))
	
while f1 < thresholdF1 and numberOfIterations < maxNoOfIterations:
    logger.info("\n")
    logger.info('Iteration Number: ' + str(numberOfIterations))
    logger.info("\n")
    #random.shuffle(auxList)
    random.shuffle(auxTweets)
    currentIterTweets = auxTweets[:20]
    #auxTweets = auxTweets[20:]
    with open('auxTweets.txt','w') as fout:
        for tweet in currentIterTweets: 
            fout.write(tweet + '\n')
    fout.close()
    #log and write to auxTweets
    
    auxiliaryDataFile = fasttext('auxTweets.txt')
    currentIterFeat = np.genfromtxt(auxiliaryDataFile)
    
    iterationLabels = []
    iterationTweets = []
    positiveFeat = []
    negativeFeat = []
    count = 0
    for tweetFeature in currentIterFeat:
        meancosineSim = cosineSimilarity(tweetFeature, positiveTrainingFeatures) 
        #print(meancosineSim)
        #print(tweetFeature)
        #print(meancosineSim)
        if(meancosineSim > minSimilarityThreshold):
            iterationLabels.append(1.0)
            iterationTweets.append(tweetFeature)
            positiveFeat.append(tweetFeature)
            logger.info(str(1) + ' ' +  str(meancosineSim) + ' ' + currentIterTweets[count])
            #print('yes')
            #print(iterationLabels)
        elif meancosineSim > minSimilarityThreshold - similarityWindowSize:
            #print('pass')
            logger.info(str(-1) + ' ' +  str(meancosineSim) +' ' + currentIterTweets[count])
            pass
        else:
            iterationLabels.append(0.0)
            iterationTweets.append(tweetFeature)
            negativeFeat.append(tweetFeature)
            logger.info(str(0) + ' ' + str(meancosineSim) + ' ' + currentIterTweets[count])
            # print('no')
            # print(iterationLabels)
        count+=1

	# print(iterationTweets)
	# print(iterationLabels)
	
    iterationTrainingFeatures = [] 
    for feature in trainFeatures:
        iterationTrainingFeatures.append(feature)
    for feature in iterationTweets:
        iterationTrainingFeatures.append(feature)
		
    iterationTrainingLabels = []
    for label in trainLabels:
        iterationTrainingLabels.append(label)
    for label in iterationLabels:
        iterationTrainingLabels.append(label)
	# print('len of feat: ')
	# print(len(iterationTrainingFeatures))
	# print('len of labels: ')
	# print(len(iterationTrainingLabels))
    print(len(iterationTrainingFeatures))
    print(len(iterationTrainingLabels))
    iterationF1 = SLP(iterationTrainingFeatures, iterationTrainingLabels, validationFeatures, validationLabels) 
    logger.info(str(iterationF1))
    numberOfIterations += 1
    print('iterationF1: ')
    print(iterationF1)
	
    if ((iterationF1 - f1) > auxThresholdexpectation):
        for feat in positiveFeat:
            positiveTrainingFeatures.append(feat)
        auxTweets = auxTweets[20:]
        f1 = iterationF1
        trainFeatures = iterationTrainingFeatures
        trainLabels = iterationTrainingLabels
        bestIteration = numberOfIterations
				
logger.info('Initial F1 : ' + str(initial_F1))
print('F1 score : ')
print(f1)
print('Best iteration so far : ')
print(bestIteration)
logger.info('Max F1 : ' + str(f1))
logger.info('Best Iteration : ' + str(bestIteration))

	
			

