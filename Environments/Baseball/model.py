###########################################################################
# Code by: Will Melville
# (Minor changes made for incorporation into the skill estimation framework)
###########################################################################

import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import json,code
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import sys,os,argparse
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split("model.py")[0]

for each in ["data","utilsBaseball"]:
	module = SourceFileLoader(each,f"{mainFolderName}{each}.py").load_module()
	sys.modules[each] = module


# Network hyperparameters 
VEC_SIZE = 13


class BatterPitch2Vec(nn.Module):
	
	def __init__(self,batter_indices,xswing_feats):
		super().__init__()
		self.batter_embedding = nn.Embedding(num_embeddings = batter_indices.shape[0], embedding_dim = VEC_SIZE)
		self.input_layer = nn.Linear(in_features = VEC_SIZE + len(xswing_feats) - 1, out_features = 250)
		self.hidden_layer = nn.Linear(in_features = 250, out_features= 125)
		self.hidden_layer1 = nn.Linear(in_features = 125, out_features = 64)
		self.hidden_layer2 = nn.Linear(in_features = 64, out_features = 32)
		self.output_layer = nn.Linear(in_features = 32, out_features = 9)
		self.relu = nn.ReLU()
		
	def forward(self, x):

		# x has a column for every entry in xswing_feats. 
		# The last column is the batter, so that's what we would feed 
		# into the batter embedding. 
		batter_idx = x[:,x.shape[1] - 1]
		batter_idx = batter_idx.int()
		batter_embedding = self.batter_embedding(batter_idx)
		
		# Concatenate the embedding with the rest of x 
		concatenated = torch.cat((batter_embedding, x[:,:x.shape[1] - 1]),1)
		output = self.relu(self.input_layer(concatenated))
		output = self.relu(self.hidden_layer(output))
		output = self.relu(self.hidden_layer1(output))
		output = self.relu(self.hidden_layer2(output))

		# Don't need to apply softmax because the loss function 
		# crossentropyloss does that for us
		output = self.output_layer(output) 
		
		return output


def trainModel(learningRate,epochs,batterIndices,train,test):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	trainloader,testloader = sys.modules["data"].convertToTorch(train,test)

	model = BatterPitch2Vec(batterIndices,sys.modules["data"].xswingFeats).to(device)

	# The initial weights for the batter has a big effect on the embedding, so initialize them all to 0 
	model.batter_embedding.weight.data.fill_(1.)
	optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
	loss_fn = nn.CrossEntropyLoss()


	# Training: Loops 
	trainLosses = []
	testLosses = []
	trainAccs = []
	testAccs = []

	for i in range(epochs):
		# Tell the model that we're training now 
		model.train()
		
		# Loop through batches of training data 
		loss_val = 0
		num_correct =  0
		
		for batch, (x, y_truth) in enumerate(trainloader):
			x = x.to(device)
			y_truth = y_truth.to(device)
			
			# Zero out gradients 
			optimizer.zero_grad()
			
			# Run the model 
			outputs = model(x)
			
			# Calculate loss 
			loss = loss_fn(outputs, y_truth)
			loss_val += loss.item()
			
			# Acc
			predicted = outputs.cpu().detach().numpy().argmax(axis = 1)
			num_correct += np.sum(predicted == y_truth.cpu().numpy())
			
			# Backprop 
			loss.backward()
			optimizer.step()
			
			
		trainLosses.append(loss_val / len(trainloader))
		trainAccs.append(num_correct / (len(trainloader) * 128))
		
		model.eval()
		loss_val = 0
		num_correct = 0
		for batch, (x, y_truth) in enumerate(testloader):
			x = x.to(device)
			y_truth = y_truth.to(device)
			preds = model(x)
			
			# Loss
			loss = loss_fn(preds, y_truth)
			loss_val += loss.item()
			# Acc
			predicted = preds.cpu().detach().numpy().argmax(axis = 1)
			num_correct += np.sum(predicted == y_truth.cpu().numpy())
		   
		
		testLosses.append(loss_val/ len(testloader))
		testAccs.append(num_correct / (len(testloader) * 128))
		
		print('epoch : {}/{}, loss = {:.6f}'.format(i + 1, epochs, trainLosses[-1]))

	return model,trainLosses,testLosses,trainAccs,testAccs


def getModel(learningRate,epochs,batterIndices,train,test,withinFramework=False):

	modelFileName = f"model-learningRate{learningRate}-epochs-{epochs}"

	if withinFramework:
		modelFolder = f"Environments{os.sep}Baseball{os.sep}"	
	else:
		modelFolder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}Models{os.sep}"	


	for folder in [modelFolder, f"{modelFolder}imgs{os.sep}"]:
		# If the folder doesn't exist already, create it
		if not Path(folder).is_dir():
			os.mkdir(folder)


	# Load model and info, if present
	if Path(f"{modelFolder}{modelFileName}").is_file():
		
		print(f"\nLoading existing trained model... (Learning Rate: {learningRate}) | Epochs: {epochs}")
		# code.interact("...", local=dict(globals(), **locals()))


		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		if withinFramework:
			model = sys.modules["model"].BatterPitch2Vec(batterIndices,sys.modules["data"].xswingFeats).to(device)
		else:
			model = BatterPitch2Vec(batterIndices,sys.modules["data"].xswingFeats).to(device)
		
		model.load_state_dict(torch.load(f"{modelFolder}{modelFileName}"))
		model.eval()


		with open(f"{modelFolder}info-{modelFileName}.json") as infile:
			results = json.load(infile)
			trainLosses = results["trainLosses"]
			testLosses = results["testLosses"]
			trainAccs = results["trainAccs"]
			testAccs = results["testAccs"]
		
		print(f"Model was loaded successfully")
	
	# Otherwise, proceed with model creation & training
	else:
		# Since assuming model needs to be present (trained) already
		# To use within estimation framework
		if withinFramework:
			print("\nModel not found. Please run script to train the model first. Once trained, place it inside 'Environments/Baseball'.")
			exit()
		else:	
			print(f"\nTraining model... (Learning Rate: {learningRate}) | Epochs: {epochs}")
			model,trainLosses,testLosses,trainAccs,testAccs = trainModel(learningRate,epochs,batterIndices,train,test)

			# Save model
			torch.save(model.state_dict(),f"{modelFolder}{modelFileName}")

			# Save info
			results = {}
			results["trainLosses"] = trainLosses
			results["testLosses"] = testLosses
			results["trainAccs"] = trainAccs
			results["testAccs"] = testAccs
			
			with open(f"{modelFolder}info-{modelFileName}.json","w") as outfile:
				json.dump(results,outfile)

			print(f"Model was saved successfully")
		
	return model,trainLosses,testLosses,trainAccs,testAccs,modelFolder


def main():
	
	# Get arguments from command line
	parser = argparse.ArgumentParser(description="Obtain statcast data for given date range")
	
	parser.add_argument("-startYear1", dest = "startYear1", help = "Desired start year for 1st set of data.", type = str, default = "2021")
	parser.add_argument("-endYear1", dest = "endYear1", help = "Desired end year for 1st set of data.", type = str, default = "2021")
	
	parser.add_argument("-startMonth1", dest = "startMonth1", help = "Desired start month for 1st set of data.", type = str, default = "01")
	parser.add_argument("-endMonth1", dest = "endMonth1", help = "Desired end month for 1st set of data.", type = str, default = "12")
	
	parser.add_argument("-startDay1", dest = "startDay1", help = "Desired start day for 1st set of data.", type = str, default = "01")
	parser.add_argument("-endDay1", dest = "endDay1", help = "Desired end day for 1st set of data.", type = str, default = "31")
	

	parser.add_argument("-startYear2", dest = "startYear2", help = "Desired start year for 2nd set of data.", type = str, default = "2022")
	parser.add_argument("-endYear2", dest = "endYear2", help = "Desired end year for 2nd set of data.", type = str, default = "2022")
	
	parser.add_argument("-startMonth2", dest = "startMonth2", help = "Desired start month for 2nd set of data.", type = str, default = "01")
	parser.add_argument("-endMonth2", dest = "endMonth2", help = "Desired end month for 2nd set of data.", type = str, default = "12")
	
	parser.add_argument("-startDay2", dest = "startDay2", help = "Desired start day for 2nd set of data.", type = str, default = "01")
	parser.add_argument("-endDay2", dest = "endDay2", help = "Desired end day for 2nd set of data.", type = str, default = "31")

	args = parser.parse_args()



	###########################################################################
	# Initial Setup
	###########################################################################
	
	folders = [f"..{os.sep}..{os.sep}Data{os.sep}",
			  f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}",
			  f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}Models{os.sep}",
			  f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}Models{os.sep}imgs"]
	
	# If the folder doesn't exist already, create it
	for folder in folders:
		if not Path(folder).is_dir():
			os.mkdir(folder)

	###########################################################################



	###########################################################################
	# Data
	###########################################################################

	rawData1 = sys.modules["data"].getData(args.startYear1,args.startMonth1,args.startDay1,args.endYear1,args.endMonth1,args.endDay1)
	rawData2 = sys.modules["data"].getData(args.startYear2,args.startMonth2,args.startDay2,args.endYear2,args.endMonth2,args.endDay2)

	train,test,allData,batterIndices = sys.modules["data"].manageDataForModel(rawData1,rawData2)

	###########################################################################
	


	###########################################################################
	# MODEL
	###########################################################################

	# Set hyperparameters
	learningRate = 1e-5
	epochs = 20 #40 #might be worth it to do like 40 epochs or more.

	model,trainLosses,testLosses,trainAccs,testAccs,modelFolder = \
					getModel(learningRate,epochs,batterIndices,train,test)

	###########################################################################
	


	###########################################################################
	# PLOTS
	###########################################################################
	
	plt.plot(trainLosses,label='Train')
	plt.plot(testLosses,label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(f"{modelFolder}imgs{os.sep}model-learningRate{learningRate}-epochs-{epochs}-EpochsVsLoss.jpg")
	plt.clf()

	plt.plot(trainAccs, label='Train')
	plt.plot(testAccs, label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(f"{modelFolder}imgs{os.sep}model-learningRate{learningRate}-epochs-{epochs}-EpochsVsAccuracy.jpg")
	plt.clf()

	###########################################################################


	###########################################################################
	# UTILITY FUNCTION
	###########################################################################

	# Example of calculating utilities 
	import warnings
	warnings.filterwarnings('ignore')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	test_preds = nn.functional.softmax(model(torch.tensor(test[sys.modules["data"].xswingFeats].values.astype(float), dtype = torch.float32).to(device)), dim = 1).cpu().detach().numpy()

	# test[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']] = test_preds

	for i in range(9):
		test[f'o{i}'] = test_preds[:,i]


	testWithUtilities = sys.modules["utilsBaseball"].getUtilities(test)

	testWithUtilities[['player_name', 'pitch_type', 'balls', 'strikes', 'take_utility', 'swing_utility']].head()

	###########################################################################
	

if __name__ == '__main__':
	main()