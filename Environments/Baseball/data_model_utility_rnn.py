#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
import torch
import torch.nn as nn 
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn.functional import log_softmax, pad
import copy,os
import gc
import torch.optim as optim
from tqdm import tqdm
from IPython.core.ultratb import AutoFormattedTB
__ITB__ = AutoFormattedTB(mode = 'Verbose',color_scheme='LightBg', tb_offset = 1)


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[6]:


#function to clean and organize data from statcast 
def organize_data(df):

    df = df.loc[df.balls < 4]
    df = df.loc[df.strikes < 3]

    df.dropna(subset = ['release_extension', 
                      'release_speed','release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z'], axis = 0,
            inplace = True)

    #convert movement to inches instead of feet 
    df[['mx', 'mz']] = df[['pfx_x', 'pfx_z']].values * 12

    #one hot encode handedness
    pit_hand = pd.get_dummies(df['p_throws'], drop_first = False)
    bat_hand = pd.get_dummies(df['stand'], drop_first = False)
    df['pit_handR'] = pit_hand['R']
    df['bat_handR'] = bat_hand['R']
    df = df.drop(['p_throws', 'stand', 'pfx_x', 'pfx_z'], axis = 1)
    
    #remove bunts 
    df = df.loc[df.description.isin(['foul_bunt', 'bunt_foul_tip', 'missed_bunt']) == False]
    df = df.loc[df.events != 'sac_bunt']

    #define the pitch outcome 
    df['outcome'] = -1
    df.loc[df.type == 'B', 'outcome'] = 0 #called ball 
    df.loc[df.description == 'called_strike', 'outcome'] = 1 #called strike 
    df.loc[df.description.isin(['swinging_strike', 'swinging_strike_blocked']), 'outcome'] = 2 #swm 
    df.loc[df.description.isin(['foul', 'foul_tip']), 'outcome'] = 3 #foul ball 

    #the other outcomes are all batted balls, which should either be outs or singles, doubles, triples, or home runs 
    df.loc[(df.type == 'X') & (df.events.isin(['field_out', 'force_out', 'field_error', 'grounded_into_double_play', 'sac_fly', 'fielders_choice', 
                                               'fielders_choice_out', 'double_play', 'other_out', 'triple_play', 
                                               'sac_fly_double_play'])), 'outcome'] = 4 # in play out 
    df.loc[(df.type == 'X') & (df.events == 'single'), 'outcome'] = 5 #single 
    df.loc[(df.type == 'X') & (df.events == 'double'), 'outcome'] = 6 # double 
    df.loc[(df.type == 'X') & (df.events == 'triple'), 'outcome'] = 7 #triple 
    df.loc[(df.type == 'X') & (df.events == 'home_run'), 'outcome'] = 8 #hr 

    #if outcome is still -1, drop it 
    df = df.loc[df.outcome != -1]

    #define an is_swing column 
    df['is_swing'] = -1 
    df.loc[df.description.isin(['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip']), 'is_swing'] = 1
    df.loc[df.description.isin(['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch', 'pitchout']), 'is_swing'] = 0

    #define an is_miss column 
    df['is_miss'] = -1 
    df.loc[df.is_swing == 0 , 'is_miss'] = 0
    df.loc[df.description.isin(['swinging_strike', 'swinging_strike_blocked']), 'is_miss'] = 1 
    df.loc[df.description.isin(['hit_into_play', 'foul', 'foul_tip']), 'is_miss'] = 0
    return df


# In[7]:


#columns needed from statcast 
needed_columns = ['game_date', 'game_year', 'game_pk', 'player_name', 'pitcher', 'batter', 'pitch_type', 'pitch_name', 'stand', 'p_throws', 'balls', 'strikes', 'release_speed', 
                  'release_spin_rate', 'release_extension', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
                  'plate_x', 'plate_z',  'type', 'events', 'description', 'woba_value', 'at_bat_number', 'pitch_number']


# In[8]:


#from google.colab import drive
#drive.mount('/drive')

folder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"

#read in the data
raw22 = pd.read_csv(f'{folder}{os.sep}raw22.csv')
raw21 = pd.read_csv(f'{folder}{os.sep}raw21.csv')
raw19 = pd.read_csv(f'{folder}{os.sep}raw18_19_20.csv')
raw21 = raw21[needed_columns]
raw22 = raw22[needed_columns]
raw19 = raw19[needed_columns]


# In[9]:


#clean the data

df21 = organize_data(raw21)
df22 = organize_data(raw22)
df19 = organize_data(raw19)


all_data = df22.append(df21, ignore_index = True)
all_data = all_data.append(df19, ignore_index = True)


# In[10]:


#min max scale variables 
standardizer = StandardScaler().fit(all_data[['release_speed', 'mx', 'mz', 
     'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
     'release_pos_x', 'release_pos_z']].values)

all_data[['release_speed', 'mx', 'mz', 
     'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
     'release_pos_x', 'release_pos_z']] = standardizer.transform(all_data[['release_speed', 'mx', 'mz', 
     'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
     'release_pos_x', 'release_pos_z']].values)


# In[11]:


#get the batter index
batter_indices = pd.DataFrame({'batter': all_data.batter.unique()})

batter_indices['batter_index'] = batter_indices.index.values

#merge 
all_data = all_data.merge(batter_indices, on = 'batter')


# **Data Loader**
# 
# For the RNN, you have to load in pitches one plate appearance at a time, so this next code cell stores batches of plate appearances in a list to hopefully make running easier

# In[12]:


train = all_data.copy()


# In[13]:


batch_size = 256

#define a dataloader type thing. Basically we break down at bats by pitch sequence length. 
#then for each length we need to break down all the plate appearances of that length into batches.
pa_lengths = train.groupby(['game_pk', 'at_bat_number'], as_index = False).agg(seq_length = ('pitch_number', 'max'),
                                                                               pitches_recorded = ('pitch_number', 'count'))
#merge with train
train = train.merge(pa_lengths, how = 'inner', on = ['game_pk', 'at_bat_number'])
#make sure the pitches recorded matches the sequence length
train = train.loc[train.seq_length == train.pitches_recorded]

train['pa_id'] = [str(train.game_pk.values[i]) + '-' + str(train.at_bat_number.values[i]) for i in range(train.shape[0])]
#train is ordered from latest to earliest, but I think we want our sequence to go from earliest up top to latest on bottom 
train = train[::-1]

features = ['balls', 'strikes', 'release_speed', 'release_spin_rate', 'release_extension', 'plate_x', 'plate_z',
                'mx', 'mz', 'pit_handR', 'bat_handR', 'batter_index']

loader_list = []
for l in pa_lengths.seq_length.unique():
  d = train.loc[train.seq_length == l]
  pas = list(d.pa_id.unique())
  #for each batch, I need to continue to add plate appearances until I hit my batch size 
  while len(pas) > 0:
    pas_for_batch = []
    s = 0 
    while s < batch_size:
      if len(pas) != 0:
        pas_for_batch.append(pas.pop(0))
        s += l
      else:
        s+=1

    #get the pas for this batch 
    batch_data = d.loc[d.pa_id.isin(pas_for_batch)]
    pa_id_and_pitch_num = batch_data[['pa_id', 'pitch_number']].values
    batch_x = batch_data[features].values
    batch_y = batch_data.outcome.values.astype(int)
    #reshape so that each pa is a separate entry in the batch
    batch_x = batch_x.reshape((len(pas_for_batch), l, len(features)))
    batch_y = batch_y.reshape((len(pas_for_batch),l))
    torch_batch_x = torch.tensor(batch_x, dtype = torch.float)
    torch_batch_y = torch.tensor(batch_y, dtype = torch.long)
    
    #append to list 
    loader_list.append((pa_id_and_pitch_num, torch_batch_x, torch_batch_y))


#shuffle the batches so that the rnn is trained on variable length sequences right from the get go 
#this will also give you a train test split 
import random
random.seed(10)
random.shuffle(loader_list)
train_loader_list = loader_list[:2*len(loader_list)//3]
test_loader_list = loader_list[2*len(loader_list)//3:]


# **RNN**

# In[14]:


#helper classes 

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


# In[15]:


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


# **Helper Functions**

# In[16]:


def prediction_func(input, target, recurrent = True):
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


# **Predictions**

# In[17]:


model = RNN(hidden_size = 32, output_size = 9).to(device)
model.load_state_dict(torch.load('final_OP',map_location=device))


# In[19]:


loader_list = loader_list[:10]


# In[22]:


#code to run model on all data. This will take I think around 30-45 minutes. It takes 10-15 minutes if you just use the test_loader_list 
predicted_data = pd.DataFrame()
for batch in range(len(loader_list)):
  ids, x, y = loader_list[batch]
  
  ypred = prediction_func(x,y)

  y = y.view(y.size(0)*y.size(1))
  losses = nn.CrossEntropyLoss(reduction = 'none')(ypred, y.to(device))
  new_data = pd.DataFrame({'pa_id': ids[:,0], 'pitch_number': ids[:,1], 'outcome': y.cpu().numpy()})
  new_data[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']] = nn.functional.softmax(ypred, dim = 1).detach().cpu().numpy()
  new_data['cross_entropy_loss'] = losses.detach().cpu().numpy()
  predicted_data = predicted_data.append(new_data, ignore_index = True)


# In[28]:


#get the balls and strikes
all_data['pa_id'] = [str(all_data.game_pk.values[i]) + '-' + str(all_data.at_bat_number.values[i]) for i in range(all_data.shape[0])]
predicted_data = predicted_data.merge(all_data[['pa_id', 'pitch_number', 'balls', 'strikes']], how = 'inner', on = ['pa_id', 'pitch_number'])


# **Utility Function**

# In[30]:


count_runs = pd.DataFrame({'balls_pre_event': [3,3,2,3,2,1,0,1,2,0,1,0], 'strikes_pre_event': [0,1,0,2,1,0,0,1,2,1,2,2],
                           'val_ball': [0.131,0.201,0.110,0.276,0.103,0.063,0.034,0.050,0.098,0.027,0.046,0.022],
                           'val_strike': [-0.070,-0.076,-0.062,-0.351,-0.071,-0.050,-0.043,-0.067,-0.252,-0.062,-0.206,-0.184],
                           'val_out': [-0.496,-0.426,-0.385,-0.350,-0.323,-0.323, -0.289,-0.273,-0.252,-0.246,-0.206,-0.184],
                           'val_single': [0.287,0.356,0.397,0.432,0.459,0.460,0.494,0.510,0.530,0.537,0.577,0.598],
                           'val_double': [0.583,0.652,0.693,0.728,0.755,0.756,0.790,0.805,0.826,0.832,0.872,0.894],
                           'val_triple': [0.861,0.930,0.971,1.006,1.033,1.034,1.068,1.083,1.104,1.110,1.150,1.172],
                           'val_hr': [1.2,1.269,1.31,1.345,1.372,1.373,1.407,1.423,1.443,1.45,1.490,1.511]})


# In[31]:


def get_utilities(pitch_df):
  '''
  pitch_df is a pandas dataframe with a column for balls, strikes, and the 
  probabilities of each outcome, named o1, o2, o3, o4, o5, o6, o7, and o8

  Returns the expected utility of a swing and of a take for that pitch.
  The batter's optimal utility for the pitch is the larger of those two values
  '''

  pitch_df['swing_utility'] = -1 
  pitch_df['take_utility'] = -1
  for balls in pitch_df.balls.unique():
    for strikes in pitch_df.strikes.unique():
      d = pitch_df.loc[(pitch_df.balls == balls) & (pitch_df.strikes == strikes)]

      #get the corresponding row from count_runs table
      count_pre = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes)]

      #get the value of a called ball 
      val0 = count_pre.val_ball.values[0]

      #get the value of a called or swinging strike 
      val12 = count_pre.val_strike.values[0]

      #value of a foul ball 
      if strikes == 2:
        #no change 
        val3 = 0 
      else:
        #value of foul is just value of a strike
        val3 = count_pre.val_strike.values[0] 

      #value of ball in play out 
      val4 = count_pre.val_out.values[0]
      #single 
      val5 = count_pre.val_single.values[0]
      #double 
      val6 = count_pre.val_double.values[0]
      #triple 
      val7 = count_pre.val_triple.values[0]
      #hr 
      val8 = count_pre.val_hr.values[0]

      #calculate utilities 
      no_swing = d.o0.values + d.o1.values 
      swing = d[['o2','o3','o4','o5','o6','o7','o8']].values.sum(axis = 1)
      take_utility = (d.o0.values / no_swing) * val0 + (d.o1.values/no_swing) * val12
      swing_utility = (d.o2.values/swing) * val12 + (d.o3.values/swing)*val3 + (d.o4.values/swing)*val4 + (d.o5.values/swing)*val5 + (d.o6.values/swing)*val6 + (d.o7.values/swing)*val7 + (d.o8.values/swing) * val8
      d.swing_utility = swing_utility 
      d.take_utility = take_utility  
      pitch_df.loc[(pitch_df.strikes == strikes) & (pitch_df.balls == balls), ['swing_utility', 'take_utility']] = d[['swing_utility', 'take_utility']].values

  return pitch_df       


# In[33]:


print(get_utilities(predicted_data).head())


# In[ ]:



