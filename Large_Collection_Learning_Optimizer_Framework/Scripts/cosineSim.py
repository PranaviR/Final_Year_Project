import numpy as np
def cosineSimilarity(tweetFeature, positiveTrainFeatures):
	np.tweetFeature = tweetFeature
	np.positiveTrainFeatures = positiveTrainFeatures
	count = 0
	sum = 0
	maxCosineSimilarity = 0
	v1 = np.tweetFeature
	for v2 in np.positiveTrainFeatures:
		x = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
		if np.isnan(x):
			x = 0
		maxCosineSimilarity = np.maximum(maxCosineSimilarity,x)
		#sum += x
		#count += 1
	#meanCosineSimilarity = sum/count
	return maxCosineSimilarity