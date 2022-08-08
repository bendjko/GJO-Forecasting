from re import L
from baseline import *
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os


class Bert_Model(nn.Module):
   def __init__(self):
       super(Bert_Model, self).__init__()
       self.bert = BertModel.from_pretrained('bert-base-uncased')
       self.out = nn.Linear(self.bert.config.hidden_size, 256)
       self.drop = nn.Dropout(0.5)
       self.act = nn.ReLU()
   def forward(self, input):
       _, output = self.bert(**input, return_dict = False)
       out = self.out(output)
       out = self.act(out)
       out = self.drop(out)
       return out[0]

# LSTM input: # of forecast, forecast length(516 in our case), 
class LSTMlayer(nn.Module):
    def __init__(self, input_size=50, hidden_size=256):
        super(LSTMlayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(self.lstm.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        output, _ = self.lstm(input)
        out = self.out(output)
        out = self.sigmoid(out)
        return out

def preprocess(text):
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   encoding = tokenizer.encode_plus(
    text, 
    add_special_tokens = True, 
    truncation = True, 
    padding = "max_length", 
    return_token_type_ids = False,
    return_attention_mask = True, 
    return_tensors = "pt"
    )
   return encoding

# figure out new way to represent prediction since number of prediction varies
# for now, replicate Saketh's representation 
def binary_prediction_representation(pred):
    return pred[0]

def binary_flag(question_called, forecast_made):
    if question_called == forecast_made:
        return [1]
    else:
        return [0]

# return concatenated tensor of forecast
def binary_option_forecast_representation(day, individual_prediction):
    clf = Bert_Model()
    justification = clf(preprocess(individual_prediction["text"])).detach().numpy()
    prediction = int(binary_prediction_representation(individual_prediction["pred"]))
    flag = binary_flag(day, individual_prediction["days_past"])
    return torch.from_numpy(np.concatenate((prediction, justification, flag), axis=None))

def question_representation(dataframe):
    clf = Bert_Model()
    dim_matcher = [0, 0]
    output = clf(preprocess(dataframe.loc["title"][0])).detach().numpy()
    return torch.from_numpy(np.concatenate((output, dim_matcher), axis=None))

def binary_option_input(question, forecast):
    return torch.cat((question, forecast), 0)

def call_questions(dataframe):
    correct_answer, possible_answers = correct_possible_answer(dataframe)
    prediction_stack = get_prediction_stack(dataframe)
    longest_day = prediction_stack["days_past"].max()

# sequence, timestep, feature
# sequence: number of forecasts
# timestep: 516
# feature: 

def concatenate_tensors(tensor1, tensor2):
    # tensor1 = tensor1.detach().numpy()
    # tensor2=tensor2.detach().numpy()
    return np.vstack((tensor1, tensor2))

def return_input_tensor(preds):
    input_tensor = torch.empty(516)
    for ind in preds.index:
        pred = preds.iloc[ind]
        forecast_rep = binary_option_forecast_representation(1,pred)
        question_and_forecast = binary_option_input(question_rep, forecast_rep)
        input_tensor = concatenate_tensors(input_tensor, question_and_forecast)
        print(input_tensor)
    return torch.from_numpy(input_tensor)

test_file = os.path.expanduser("~/Desktop/question_1951.json")
dataframe = df(test_file)
question_rep = question_representation(dataframe)
preds = get_prediction_stack(dataframe)
input_tensor = return_input_tensor(preds)

print("input_tensor:", input_tensor)

# lstmlayer = LSTMlayer()
# print(lstmlayer(input_tensor))


# outline
# concatenate input forecast sequence into 3d tensor
# remove first tensor (randomized tensor) and reconfirm the size 
# using .view from torch, feed a sequence to lstm layer
# 
