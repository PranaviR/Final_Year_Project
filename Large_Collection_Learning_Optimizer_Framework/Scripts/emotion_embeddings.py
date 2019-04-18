import os
os.environ['KERAS_BACKEND'] = 'theano'
from emotion_predictor import EmotionPredictor
import keras

def get_emotion_embeddings(input_file):
    model = EmotionPredictor(classification='plutchik', setting='mc', use_unison_model=True)
    f = open(input_file)
    tweets = f.read().splitlines()
    embeddings = model.embedd(tweets)
    return embeddings

