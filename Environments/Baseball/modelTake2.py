###########################################################################
# Code by: Will Melville
# (Minor changes made for incorporation into the skill estimation framework)
###########################################################################

import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import json,code
import sys,os
import copy
from pathlib import Path
from importlib.machinery import SourceFileLoader


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split("modelTake2.py")[0]

for each in ["dataTake2","utilsBaseball"]:
	module = SourceFileLoader(each,f"{mainFolderName}{each}.py").load_module()
	sys.modules[each] = module


global batter_indices

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


features = ['balls', 'strikes', 'release_speed', 'release_spin_rate', 'release_extension', 'plate_x', 'plate_z',
			'mx', 'mz', 'pit_handR', 'bat_handR', 'batter_index']


# Helper classes 

batter_embedding_dim = 13
output_embedding_dim = 13

class InputEmbedding(nn.Module):

	def __init__(self, hidden_size = 32, output_size = 9):
		super(InputEmbedding, self).__init__()
		self.hidden_size = hidden_size 
		self.output_size = output_size
		#define batter embedding 
		self.batter_embedding = nn.Embedding(batter_indices.shape[0], embedding_dim = batter_embedding_dim)

		#define embedding for the pitch concatenated with the batter embedding concatenated with hidden state 
		self.ie1 = nn.Linear(in_features = len(features) - 1 + batter_embedding_dim + self.hidden_size, out_features = 512)
		self.ie2 = nn.Linear(512, 256)
		self.ie3 = nn.Linear(256, 128)
		self.ie4 = nn.Linear(128, 64)
		self.ie5 = nn.Linear(64, self.output_size)
		self.relu = nn.ReLU()
	
	def forward(self, x, hidden):
		#batter index is last column of x 
		batter_idx = x[:,-1].int()
		bat_emb = self.batter_embedding(batter_idx)

		#concatenate batter embedding and input and hidden state
		conc = torch.concat((x[:,:-1], bat_emb, hidden), dim = 1)
		
		#run through layers 
		conc = self.relu(self.ie1(conc))
		conc = self.relu(self.ie2(conc))
		conc = self.relu(self.ie3(conc))
		conc = self.relu(self.ie4(conc))
		return self.ie5(conc)


class HiddenStateUpdater(nn.Module):

	def __init__(self, hidden_size = 32, output_size = 9):
		super(HiddenStateUpdater, self).__init__()
		self.hidden_size = hidden_size 
		self.output_size = output_size
		#we need a target embedding 
		self.target_embedding = nn.Embedding(self.output_size, output_embedding_dim)
		#remember, we are going to take our pitch and our output embedding and concatenate them together for the hidden state, so we need features 
		#plus output embedding dim minus 1 for the batter index which we don't want to include.
		self.l1 = nn.Linear(len(features) - 1 + output_embedding_dim, 256)
		self.l2 = nn.Linear(256, 128)
		self.l3 = nn.Linear(128, 64)
		self.l4 = nn.Linear(64,32)
		self.proj = nn.Linear(32, self.hidden_size)
		self.relu = nn.ReLU()

	def forward(self, x, y):
		#get the output embedding 
		y = self.target_embedding(y)
		#concatenate onto x 
		x = torch.concat((x, y), dim = -1)
		#feed through network
		x = self.relu(self.l1(x))
		x = self.relu(self.l2(x)) 
		x = self.relu(self.l3(x))
		x = self.relu(self.l4(x))
		return self.proj(x)


class RNN(nn.Module):

	def __init__(self, hidden_size = 32, output_size = 9):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size 
		self.output_size = output_size 

		self.input_embedding = InputEmbedding(self.hidden_size, self.output_size)
		self.h_update = HiddenStateUpdater(self.hidden_size, self.output_size)
		self.relu = nn.ReLU()

		#initialize batter embeddings 
		self.input_embedding.batter_embedding.weight.data.fill_(1.)


	def forward(self, x, hidden, y):
		#first we run x and hidden through the 'input embedding'
		output = self.input_embedding(x, hidden)

		#next, we run x and y through the hidden state updater, and we add that onto the old hidden state
		hidden += self.h_update(x[:,:-1], y)

		#return x and hidden 
		return output, hidden 


	def init_hidden(self):
		return torch.zeros(1, self.hidden_size) 

  
def prediction_func(model, input, target, recurrent = True):
	#input is a batch of PAs 
	with torch.no_grad():
		#init 
		input = input.to(device)
		target = target.to(device)
		loss = 0

		input_length = input.size(1)

		#loop through pas in batch 
		preds = torch.empty((1,9)).to(device)
		for pa_num in range(input.size(0)):
			#initialize hidden state at start of each pa 
			h0 = model.init_hidden()
			h0 = h0.to(device)
			pa = input[pa_num]
			pa_targets = target[pa_num]
			for pitch in range(input_length):
				yhat, h1 = model(pa[pitch].unsqueeze(0), h0, pa_targets[pitch].unsqueeze(0))
				preds = torch.cat((preds, yhat), dim = 0)
				if recurrent:
					h0 = h1
				else:
					h0 = model.init_hidden().to(device)

		#return predictions
		return preds[1:,:]  


def main():

	global batter_indices
	

	###############################################################
	# DATA
	###############################################################

	folder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"
	csvFiles = [f"{folder}raw22.csv",f"{folder}raw21.csv",f"{folder}raw18_19_20.csv"]

	all_data,batter_indices,standardizer = sys.modules["dataTake2"].manageData(csvFiles)


	train = all_data.copy()


	batch_size = 256

	# Define a dataloader type thing. Basically we break down at bats by pitch sequence length. 
	# Then for each length we need to break down all the plate appearances of that length into batches.
	pa_lengths = train.groupby(['game_pk', 'at_bat_number'], as_index = False).agg(seq_length = ('pitch_number', 'max'),
																				   pitches_recorded = ('pitch_number', 'count'))
	# Merge with train
	train = train.merge(pa_lengths, how = 'inner', on = ['game_pk', 'at_bat_number'])
	# Make sure the pitches recorded matches the sequence length
	train = train.loc[train.seq_length == train.pitches_recorded]

	train['pa_id'] = [str(train.game_pk.values[i]) + '-' + str(train.at_bat_number.values[i]) for i in range(train.shape[0])]
	# Train is ordered from latest to earliest, but I think we want our sequence to go from earliest up top to latest on bottom 
	train = train[::-1]


	loader_list = []
	for l in pa_lengths.seq_length.unique():
		d = train.loc[train.seq_length == l]
		pas = list(d.pa_id.unique())
		
		# For each batch, I need to continue to add plate appearances until I hit my batch size 
		while len(pas) > 0:
			pas_for_batch = []
			s = 0 
			while s < batch_size:
				if len(pas) != 0:
					pas_for_batch.append(pas.pop(0))
					s += l
				else:
					s += 1

			# Get the pas for this batch 
			batch_data = d.loc[d.pa_id.isin(pas_for_batch)]
			pa_id_and_pitch_num = batch_data[['pa_id', 'pitch_number']].values
			batch_x = batch_data[features].values
			batch_y = batch_data.outcome.values.astype(int)
			
			#Reshape so that each pa is a separate entry in the batch
			batch_x = batch_x.reshape((len(pas_for_batch), l, len(features)))
			batch_y = batch_y.reshape((len(pas_for_batch),l))
			torch_batch_x = torch.tensor(batch_x, dtype = torch.float)
			torch_batch_y = torch.tensor(batch_y, dtype = torch.long)
		
			# Append to list 
			loader_list.append((pa_id_and_pitch_num, torch_batch_x, torch_batch_y))


	# Shuffle the batches so that the rnn is trained on variable length sequences right from the get go 
	# This will also give you a train test split 
	import random
	random.seed(10)
	random.shuffle(loader_list)
	train_loader_list = loader_list[:2*len(loader_list)//3]
	test_loader_list = loader_list[2*len(loader_list)//3:]

	###############################################################


	###########################################################################
	# MODEL
	###########################################################################

	model = RNN(hidden_size = 32, output_size = 9).to(device)
	model.load_state_dict(torch.load('final_OP',map_location=device))

	loader_list = loader_list[:10]

	# Code to run model on all data. This will take I think around 30-45 minutes. It takes 10-15 minutes if you just use the test_loader_list 
	predicted_data = pd.DataFrame()
	
	for batch in range(len(loader_list)):
		ids, x, y = loader_list[batch]
	  
		ypred = prediction_func(model,x,y)

		y = y.view(y.size(0)*y.size(1))
		losses = nn.CrossEntropyLoss(reduction = 'none')(ypred, y.to(device))
		new_data = pd.DataFrame({'pa_id': ids[:,0], 'pitch_number': ids[:,1], 'outcome': y.cpu().numpy()})
		new_data[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']] = nn.functional.softmax(ypred, dim = 1).detach().cpu().numpy()
		new_data['cross_entropy_loss'] = losses.detach().cpu().numpy()
		predicted_data = predicted_data.append(new_data, ignore_index = True)


	# Get the balls and strikes
	all_data['pa_id'] = [str(all_data.game_pk.values[i]) + '-' + str(all_data.at_bat_number.values[i]) for i in range(all_data.shape[0])]
	predicted_data = predicted_data.merge(all_data[['pa_id', 'pitch_number', 'balls', 'strikes']], how = 'inner', on = ['pa_id', 'pitch_number'])
	

	r = sys.modules[each].getUtilities(predicted_data).head()
	print(r)

	code.interact("...", local=dict(globals(), **locals()))

	
	###########################################################################
	

	

if __name__ == '__main__':
	main()
