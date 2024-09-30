import sys, argparse
from random import choices
import numpy as np
import json
import nltk
import optuna
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy import sparse
from collections import Counter
import operator
import json
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, accuracy_score

def read_data(filename, task):
	
	data=[]
	X=[]
	Y=[]
	
	with open(filename, encoding="utf-8") as file:
		for line in file:
			data=json.loads(line.rstrip())
			label=data["label"]
			text=data["text"]
			
			# tokenize
			tokens=nltk.word_tokenize(text)
			X.append(tokens)
			if task == "classification":
				Y.append(label)  
			elif task == "regression":
				Y.append(float(label))
				
	return X, Y 




def build_features(dataX, feature_functions):
	
	""" This function featurizes the data according to the list of parameter feature_functions """
	
	data=[]
	for tokens in dataX:
		feats={}
		
		for function in feature_functions:
			feats.update(function(tokens))

		data.append(feats)
	return data



def features_to_ids(data, feature_vocab):
	
	""" 
	
	This helper function converts a dictionary of feature names to a sparse representation
 that we can fit in a scikit-learn model.  This is important because almost all feature 
 values will be 0 for most documents (note: why?), and we don't want to save them all in 
 memory.

	"""
	new_data=sparse.lil_matrix((len(data), len(feature_vocab)))
	for idx,doc in enumerate(data):
		for f in doc:
			if f in feature_vocab:
				new_data[idx,feature_vocab[f]]=doc[f]
	return new_data


def create_vocab(data, top_n=None):
	
	""" 
	
	This helper function converts a dictionary of feature names to unique numerical ids. 
	top_n limits the features to only the n most frequent features observed in the training data 
	(in terms of the number of documents that contains it).
	
	"""
	
	counts=Counter()
	for doc in data:
		for feat in doc:
			counts[feat]+=1

	feature_vocab={}

	for idx, (k, v) in enumerate(counts.most_common(top_n)):
		feature_vocab[k]=idx
				
	return feature_vocab

bestAcc=-1
bestClf=None
bestC=-1

class Objective:


	def __init__(self, trainX_ids, devX_ids, testX_ids, trainY, devY, testY, metric, task):
		self.task=task
		self.metric=metric
		self.trainX_ids=trainX_ids
		self.devX_ids=devX_ids
		self.testX_ids=testX_ids
		self.trainY=trainY
		self.devY=devY
		self.testY=testY
		self.metric=metric
		self.task=task


		if metric == "rho":
			self.metricFn=spearmanr
		elif metric == "accuracy":
			self.metricFn=accuracy_score

	def __call__(self, trial):
		

		global bestAcc, bestClf, bestC
		c=trial.suggest_float("c", 0.001, 200)

		if self.task == "classification":
			clf = linear_model.LogisticRegression(C=c, solver='lbfgs', penalty='l2', max_iter=10000)
		elif self.task == "regression":
			clf = linear_model.Ridge(alpha=c, solver='auto', max_iter=10000)

		clf.fit(self.trainX_ids, self.trainY)
		preds=clf.predict(self.devX_ids)

		score=self.metricFn(preds, self.devY)
		
		if self.metric == "rho":
			score=score[0]


		if score > bestAcc:
			bestAcc=score
			bestClf=clf
			bestC=c

		return score


def bootstrap(gold, predictions, metric, metricN, B=1000, confidence_level=0.95):
	# print(len(gold))
	critical_value=(1-confidence_level)/2
	lower_sig=100*critical_value
	upper_sig=100*(1-critical_value)
	data=[]
	for g, p in zip(gold, predictions):
		data.append([g, p])

	accuracies=[]
	
	for b in range(B):
		choice=choices(data, k=len(data))
		choice=np.array(choice)
		accuracy=metric(choice[:,0], choice[:,1])
		if metricN == "rho":
			accuracy=accuracy[0]
		
		accuracies.append(accuracy)
	
	percentiles=np.percentile(accuracies, [lower_sig, 50, upper_sig])
	
	lower=percentiles[0]
	median=percentiles[1]
	upper=percentiles[2]
	
	return lower, median, upper


def pipeline(trainX, devX, testX, trainY, devY, testY, feature_functions, task, metric):

	global bestAcc, bestClf, bestC


	""" This function evaluates a list of feature functions on the training/dev data arguments """
	
	trainX_feat=build_features(trainX, feature_functions)
	devX_feat=build_features(devX, feature_functions)
	testX_feat=build_features(testX, feature_functions)

	# just create vocabulary from features in *training* data.
	feature_vocab=create_vocab(trainX_feat, top_n=100000)

	trainX_ids=features_to_ids(trainX_feat, feature_vocab)
	devX_ids=features_to_ids(devX_feat, feature_vocab)
	testX_ids=features_to_ids(testX_feat, feature_vocab)
	

	study = optuna.create_study(directions=["maximize"], pruner=optuna.pruners.HyperbandPruner())
	study.optimize(Objective(trainX_ids, devX_ids, testX_ids, trainY, devY, testY, metric, task), n_trials=50)


	if metric == "rho":
		metricFn=spearmanr
	elif metric == "accuracy":
		metricFn=accuracy_score

	preds=bestClf.predict(testX_ids)
	lower, mid, upper=bootstrap(testY, preds, metricFn, metric)

	print("Best test Accuracy @ C=%.7f: %.3f [%.3f-%.3f]" % (bestC,mid, lower, upper))
	
	return bestClf, feature_vocab



def majority_class(trainY, devY, task):
	if task == "classification":
		labelCounts=Counter()
		for label in trainY:
			labelCounts[label]+=1
		majority_class=labelCounts.most_common(1)[0][0]
		
		correct=0.
		for label in devY:
			if label == majority_class:
				correct+=1
				
		print("Majority class:\t%s\t%.3f" % (majority_class, correct/len(devY)))

	elif task == "regression":
		avg=[np.mean(trainY)]*len(devY)
		rho, _=spearmanr(avg, devY)
		print("Mean baseline:\t%s\t%.3f" % (majority_class, rho))



def unigram_feature(tokens):
	feats={}
	for word in tokens:
		feats["UNIGRAM_%s" % word]=1
	return feats



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train regression')
	parser.add_argument('--input', '-i', type=str, help='Input folder')
	parser.add_argument('--metric', '-m', type=str, help='Evaluation metric: rho, mse, accuracy')
	parser.add_argument('--task', '-t', type=str, help='Task: classification, regression')

	args = parser.parse_args()

	print(args)

	metric=args.metric
	directory=args.input
	task=args.task


	trainX, trainY=read_data("%s/train.jsonl" % directory, task)
	devX, devY=read_data("%s/dev.jsonl" % directory, task)
	testX, testY=read_data("%s/test.jsonl" % directory, task)

	majority_class(trainY, testY)

	features=[unigram_feature]
	clf, vocab=pipeline(trainX, devX, testX, trainY, devY, testY, features, task, metric)




