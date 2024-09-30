import argparse
import datetime
import sys, json
from math import sqrt
from transformers import BertModel, BertTokenizer
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import optuna
import time
from sklearn.metrics import mean_squared_error, accuracy_score
from random import choices
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import (
	LoraConfig,
	get_peft_model,
	get_peft_model_state_dict,
	prepare_model_for_kbit_training,
)

random.seed(1)

bestOverallDev=0
testResults_for_bestOverallDev={}


metrics_bigger_is_better={"rho", "accuracy"}



def read_data(filename, task):
	"""
	:param filename: the name of the file
	:return: list of tuple ([word index list], label)
	as input for the forward and backward function
	"""
	data = []
	data_labels = []
	with open(filename) as file:
		
		for idx, line in enumerate(file):

			datum=json.loads(line)
			if task == "classification":
				label = datum["label"]
			elif task == "regression":
				label = float(datum["label"])

			text = datum["text"]

			data.append(text)
			data_labels.append(label)


	# shuffle the data
	tmp = list(zip(data, data_labels))
	random.shuffle(tmp)
	data, data_labels = zip(*tmp)

	return data, data_labels

def spearman_rho(preds, golds):
	rho, pval=spearmanr(preds, golds)
	return rho

def bootstrap(gold, predictions, metric, B=1000, confidence_level=0.95):
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
		
		accuracies.append(accuracy)
	
	percentiles=np.percentile(accuracies, [lower_sig, 50, upper_sig])
	
	lower=percentiles[0]
	median=percentiles[1]
	upper=percentiles[2]
	
	return lower, median, upper

def evaluate(model, task, all_x, all_y, metric):
	model.eval()
	corr = 0.
	total = 0.

	preds=[]
	golds=[]

	with torch.no_grad():
		idx=0
		for x, y in zip(all_x, all_y):

			idx+=1

			y_preds=model.forward(x)

			if task == "classification":
				for idx, y_pred in enumerate(y_preds):
					prediction=torch.argmax(y_pred)
					preds.append(prediction.float().cpu().numpy())
					golds.append(y[idx].float().cpu().numpy())

			elif task == "regression":
				for idx, y_pred in enumerate(y_preds):
					preds.append(y_pred[0].float().cpu().numpy())
					golds.append(y[idx].float().cpu().numpy())
				
	if metric == "rho":
		lower, median, upper=bootstrap(golds, preds, spearman_rho)
		return median, lower, upper
	elif metric == "mse":
		lower, median, upper=bootstrap(golds, preds, mean_squared_error)
		return median, lower, upper
	elif metric == "accuracy":
		lower, median, upper=bootstrap(golds, preds, accuracy_score)
		return median, lower, upper


class BERTClassifier(nn.Module):

	def __init__(self, model_path, num_labels, max_length=None, device=None, device_map=None, tokenizer_path=None):
		super().__init__()

		self.max_length=max_length
		self.device_map=device_map
		self.device=device
		self.model_name=model_path
		self.num_labels=num_labels
		self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False)
		self.bert = BertModel.from_pretrained(self.model_name)

		self.fc = nn.Linear(self.bert.config.hidden_size , self.num_labels)


	def forward(self, x):

		batch_x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)


		bert_output = self.bert(input_ids=batch_x["input_ids"],
						 attention_mask=batch_x["attention_mask"],
						 token_type_ids=batch_x["token_type_ids"],
						 output_hidden_states=True)

		bert_hidden_states = bert_output['hidden_states']

		# We're going to represent an entire document just by its [CLS] embedding (at position 0)
		out = bert_hidden_states[-1][:,0,:]

		out = self.fc(out)

		return out#.squeeze()

class BERTRegressor(nn.Module):

	def __init__(self, model_path, max_length=None, device=None, device_map=None, tokenizer_path=None):
		super().__init__()

		self.max_length=max_length
		self.device_map=device_map
		self.device=device
		self.model_name=model_path
		self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False)
		self.bert = BertModel.from_pretrained(self.model_name)

		self.fc = nn.Linear(self.bert.config.hidden_size, 1)


	def forward(self, x):

		batch_x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to(self.device)


		bert_output = self.bert(input_ids=batch_x["input_ids"],
						 attention_mask=batch_x["attention_mask"],
						 token_type_ids=batch_x["token_type_ids"],
						 output_hidden_states=True)

		bert_hidden_states = bert_output['hidden_states']

		# We're going to represent an entire document just by its [CLS] embedding (at position 0)
		out = bert_hidden_states[-1][:,0,:]

		out = self.fc(out)

		return out



def get_batches(all_x, all_y1, task, batch_size=4):

	""" Get batches for input x, y data, with data tokenized according to the BERT tokenizer
	(and limited to a maximum number of WordPiece tokens """

	batches_x=[]
	batches_y1=[]
	for i in range(0, len(all_x), batch_size):

		current_batch=[]

		batch_x = all_x[i:i+batch_size]
		batch_y1=all_y1[i:i+batch_size]
		
		batches_x.append(batch_x)

		if task == "regression":
			batches_y1.append(torch.FloatTensor(batch_y1))
		elif task == "classification":
			batches_y1.append(torch.LongTensor(batch_y1))

	return batches_x, batches_y1



def run_test(model, test_x, test_y, metric):
	start_time = datetime.datetime.now()
	

def labels2int(ys, labelset):
	intlabels=[]
	for y in ys:
		if y in labelset:
			intlabels.append(labelset[y])
		else:
			print("%s not in labels" % y)
			sys.exit(1)
	return intlabels



class Objective:


	def __init__(self, model_filename, task, max_length, device_map, device, metric, train_x, train_y, dev_x, dev_y, test_x, test_y):
		self.model_filename=model_filename
		self.task=task
		self.max_length=max_length
		self.device_map=device_map
		self.device=device
		self.metric=metric
		self.train_x=train_x
		self.train_y=train_y
		self.dev_x=dev_x
		self.dev_y=dev_y
		self.test_x=test_x
		self.test_y=test_y

	def __call__(self, trial):
		
		learningRate=trial.suggest_float("lr", 1e-6, 5e-3, log=True)

		global bestOverallDev, testResults_for_bestOverallDev

		start_time = datetime.datetime.now()

		model_path="bert-base-cased"

		if self.task == "classification":
			labelset={}
			for y in self.train_y:
				if y not in labelset:
					labelset[y]=len(labelset)			


			num_labels=len(labelset)
			train_y=labels2int(self.train_y, labelset)
			dev_y=labels2int(self.dev_y, labelset)
			test_y=labels2int(self.test_y, labelset)


			model=BERTClassifier(model_path, num_labels, max_length=self.max_length, device_map=self.device_map, device=self.device).to(self.device)
			loss_fn=nn.CrossEntropyLoss()

		elif task == "regression":
			train_y=self.train_y
			dev_y=self.dev_y
			test_y=self.test_y
			
			model=BERTRegressor(model_path, max_length=self.max_length, device_map=self.device_map, device=self.device).to(self.device)
			loss_fn=nn.MSELoss()

		batch_x, batch_y = get_batches(self.train_x, train_y, task, batch_size=32)
		dev_batch_x, dev_batch_y = get_batches(self.dev_x, dev_y, task, batch_size=32)
		test_batch_x, test_batch_y = get_batches(self.test_x, test_y, task, batch_size=32)
		
		optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

		num_epochs=100

		best_dev_acc = 0.

		if metric not in metrics_bigger_is_better:
			best_dev_acc=float('inf')

		best_epoch=0
		patience=10

		for epoch in range(num_epochs):
			model.train()

			# Train
			for x, y in tqdm(list(zip(batch_x, batch_y))):
				y_pred = model.forward(x)

				if task == "classification":
					loss = loss_fn(y_pred.view(-1, model.num_labels), y.to(self.device).view(-1))
				elif task == "regression":
					loss = loss_fn(y_pred.view(-1), y.to(self.device).view(-1))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()


			# For efficiency, evaluate on dev and test
			# But true TEST results are only for that model that corresponds to the best DEV performance
			# i.e., select model on DEV data and use that model to evaluate on TEST.

			# Evaluate
			dev_accuracy, lower, upper=evaluate(model, self.task, dev_batch_x, dev_batch_y, self.metric)
			test_accuracy, t_lower, t_upper=evaluate(model, self.task, test_batch_x, test_batch_y, self.metric)

			if (self.metric in metrics_bigger_is_better and dev_accuracy > bestOverallDev) or (self.metric not in metrics_bigger_is_better and dev_accuracy < bestOverallDev):
				bestOverallDev=dev_accuracy
				testResults_for_bestOverallDev={"test_accuracy": test_accuracy, "t_lower":t_lower, "t_upper": t_upper, "lr":learningRate, "epoch":epoch, "dev_accuracy":dev_accuracy, "d_lower": lower, "d_upper":upper, "metric": metric}
			
			print("Epoch %s\tLR: %s\tdev accuracy:\t%.3f\t[%.3f-%.3f]\ttest accuracy:\t%.3f\t[%.3f-%.3f]" % (epoch, learningRate, dev_accuracy, lower, upper, test_accuracy, t_lower, t_upper))

			if (self.metric in metrics_bigger_is_better and dev_accuracy > best_dev_acc) or (self.metric not in metrics_bigger_is_better and dev_accuracy < best_dev_acc):
				best_dev_acc = dev_accuracy
				best_epoch=epoch

			if epoch-best_epoch > patience:
				print("No change in %s epochs, exiting" % patience)
				break

			trial.report(dev_accuracy, epoch)
			if trial.should_prune():
				raise optuna.TrialPruned()

		end_time = datetime.datetime.now()
		elapsed_time = end_time - start_time

		print("LRperf\t%s\t%.3f\t%.1f" % (learningRate, best_dev_acc, elapsed_time.total_seconds()/60))

		return best_dev_acc



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train regression')
	parser.add_argument('--input', '-i', type=str, help='Input folder')
	parser.add_argument('--metric', '-m', type=str, help='Evaluation metric: rho, mse, accuracy')
	parser.add_argument('--device', '-d', type=str, help='Device: cuda:0, cuda:1, cuda:2, cuda:3, auto')
	parser.add_argument('--lr', type=str, help='Learning rate: float or "all" to sweep')
	parser.add_argument('--size', default=512, type=int, help='Max token length')
	parser.add_argument('--task', '-t', type=str, help='Task: classification, regression')

	outputModelName="bert"

	args = parser.parse_args()

	print(args)

	metric=args.metric
	directory=args.input
	deviceName=args.device
	learningRate=args.lr
	max_length=args.size
	task=args.task

	if deviceName == "auto":
		device_map="auto"
		device="cuda"

	else:
		device = torch.device(deviceName if torch.cuda.is_available() else "cpu")
		device_map=device


	print("Running on {}".format(device))

	if metric in metrics_bigger_is_better:
		bestOverallDev=0
	else:
		bestOverallDev=float('inf')


	train_x, train_y=read_data("%s/train.jsonl" % directory, task)
	dev_x, dev_y=read_data("%s/dev.jsonl" % directory, task)
	test_x, test_y=read_data("%s/test.jsonl" % directory, task)

	bestModel=None
	bestDev=0

	study = optuna.create_study(directions=["maximize"], pruner=optuna.pruners.HyperbandPruner())
	study.optimize(Objective(outputModelName, task, max_length, device_map, device, metric, train_x, train_y, dev_x, dev_y, test_x, test_y), n_trials=50)

	print("OFFICIAL TEST SCORE (MODEL SELECTION ON DEV DATA ONLY!)")
	print("OFFICIALTEST\t%s" % json.dumps(testResults_for_bestOverallDev))

