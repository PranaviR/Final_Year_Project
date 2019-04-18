import nltk
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords','english')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
input_file = 'hurricanematthew-1000.txt'
output_file = 'hm-1000.txt'

wnl = nltk.WordNetLemmatizer()

count_empty = 0
custom_stopwords = ['would', 'could','said','u']
stopWords = set(stopwords.words('english'))

#list_filtered =[]
count_empty=0	

##Tags for lemmatization
from nltk.corpus import wordnet as wn
morphy_tag = {'NN': wn.NOUN , 'JJ':wn.ADJ, 'VB':wn.VERB, 'RB':wn.ADV}



##Preprocess the text into list of lists and filter unwanted details
with open(output_file,'w') as fout:
	i = 0
	for tweet in open(input_file):
		if i%100==0:
			print(i)
		i+=1
		tweet = re.sub(r'[^\w\s]','',tweet)
		
		text = tweet.lower()
		tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
		tweet = re.sub('(\@[^\s]+)','<user>',tweet)
		try:
			tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
		except:
			pass
		word_tokens = word_tokenize(text)
		#words_filtered = []
		for w in word_tokens:
			if w.lower() not in stopWords and w not in string.punctuation and w.lower() not in custom_stopwords and w.lower()[0:4] != 'http' and not w.startswith('@'):
				pos_tuple = nltk.pos_tag([w])
				pos_tag = (pos_tuple[0][1])[:2]
				tag = morphy_tag[pos_tag] if pos_tag in morphy_tag else None 
				w = wnl.lemmatize(w, tag) if tag else wnl.lemmatize(w)
				fout.write(w+' ')
		fout.write('\n')