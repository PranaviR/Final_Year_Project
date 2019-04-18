import numpy as np
from numpy import genfromtxt
from cosineSim import cosineSimilarity
from run_fasttext import fasttext
#from augment_synonyms import augment
#from lemmatize import lemm

# from flair.models import TextClassifier
# from flair.data import Sentence
# classifier = TextClassifier.load('en-sentiment')

# positiveTweets_afteraugment = augment('positive_examples.txt', 'tweets_augment.txt')
# positiveTweets_afteraugment = 'positive_examples_augment.txt'

# positiveTweets_afterlemma = lemm(positiveTweets_afteraugment, 'positive_examples_augment_lemma.txt')
# positiveTweets_afterlemma = 'positive_examples_augment_lemma.txt'
positiveFeatures = np.genfromtxt(fasttext('positive_examples.txt'))
positiveFeatures = positiveFeatures.tolist()


##############################################################################
# positive_file = open('positive_examples.txt', 'r')
# positiveTweetList = positive_file.readlines()

# i = 0
# for tweet in positiveTweetList:
    # sentence = Sentence(tweet)
    # classifier.predict(sentence)
    # if(sentence.labels[0].value == 'POSITIVE'):
        # score = sentence.labels[0].score
    # else:
        # score = 0 - sentence.labels[0].score
    # positiveFeatures[i].append(score)
    # i += 1 
################################################################################
    
# posTweets_afteraugment = augment('tweets_positive.txt', 'tweets_augment_pos.txt')
# posTweets_afterlemma = lemm(posTweets_afteraugment, 'tweets_augment_lemma_pos.txt')
# posTweets_afterlemma = 'tweets_augment_lemma_pos.txt'
# posTweetsFeatures = np.genfromtxt(fasttext(posTweets_afterlemma))
# posTweetsFeatures = posTweetsFeatures.tolist()
posTweetsFeatures = np.genfromtxt(fasttext('tweets_positive.txt'))
posTweetsFeatures = posTweetsFeatures.tolist()


###################################################################################33
# allTweets_file = open('tweets_negative.txt', 'r')
# allTweetsList = allTweets_file.readlines()

# i = 0
# for tweet in allTweetsList:
    # sentence = Sentence(tweet)
    # classifier.predict(sentence)
    # if(sentence.labels[0].value == 'POSITIVE'):
        # score = sentence.labels[0].score
    # else:
        # score = 0 - sentence.labels[0].score
    # allTweetsFeatures[i].append(score)
    # i += 1 

for tweet in posTweetsFeatures:
    print(cosineSimilarity(tweet,positiveFeatures))
#######################################################################################3
    
#negTweets_afteraugment = augment('tweets_negative.txt', 'tweets_augment_neg.txt')
#negTweets_afteraugment = 'tweets_augment_neg.txt'
#negTweets_afterlemma = lemm(negTweets_afteraugment, 'tweets_augment_lemma_neg.txt')
#print('6')
#negTweetsFeatures = np.genfromtxt(fasttext(negTweets_afterlemma))
negTweetsFeatures = np.genfromtxt(fasttext('tweets_negative.txt'))
negTweetsFeatures = negTweetsFeatures.tolist()

for tweet in negTweetsFeatures:
	print(cosineSimilarity(tweet,positiveFeatures))
