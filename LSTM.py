# from __future__ import division, print_function
# import numpy as np
# import torch
# import torch.nn as nn
# import time
# import torch.nn.functional as F
# from torch.autograd import Variable
# from sklearn.metrics import accuracy_score
# data_path = "D:\github\mawilab20180401\mawilab\\20180401_new_10w_20_30.tsv.npy"
# label_path = "D:\github\mawilab20180401\mawilab\label_20180401_new_10w_20_30.csv.npy"
#
# print("data loading")
# data = np.load(data_path)
# labels = np.load(label_path)
#
#
# train_size = 90000
# test_size = 9999
# from sklearn.preprocessing import StandardScaler
# s1 = StandardScaler()
# s1.fit(data)
# data = s1.transform(data)
# data = data.astype(np.float32)
# labels = labels.astype(np.int64)
#
# begin = time.time()
# data_train = data[0:train_size, :]
# data_test = data[train_size:train_size + test_size, :]
# label_train = labels[0:train_size]
# label_test = labels[train_size:train_size + test_size]
# print("loading done")
# print(label_train.shape)
# print(data_train.shape)
#
#
# lstm = nn.LSTM(90,90)
# input = torch.from_numpy(data_train)
# print(input.shape)
# input.view(-1,64,90)
# print(input.shape)
# h0 = torch.randn(1,64,90)
# c0 = torch.randn(1,64,90)
# output,(hn,cn) = rnn(input,(h0,c0))
# print(output)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# def prepare_sequence(seq,to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs,dtype=torch.long)
#
# training_data = [
#     ("The dog ate the apple".split(),["DET","NN","V","DET","NN"]),
#     ("Everybody read that book".split(),["NN","V","DET","NN"])
# ]
# word_to_ix = {}
# for sent,tags in training_data:
#     for word in sent:
#         if(word not in word_to_ix):
#             word_to_ix[word] = len(word_to_ix)
#
# print(word_to_ix)
#
# EMBEDDING_DIM = 6
# HIDDEN_DIM = 6


class LSTMTagger(nn.Module):
    def __init__(self):
        super(LSTMTagger,self).__init__()
        self.lstm = nn.LSTM(90,30)
        self.hidden2tag = nn.Linear(30,2)

    def forward(self,x):
        x = x.view(-1,1,90)
        lstm_out,_ = self.lstm(x)
        return self.hidden2tag(lstm_out.view(-1,30))
