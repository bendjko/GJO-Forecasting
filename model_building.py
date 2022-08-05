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

def preprocess_justification(text):
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
    justification = Bert_Model(str(individual_prediction.loc["text"]))
    prediction = int(binary_prediction_representation(individual_prediction.loc["pred"]))
    flag = int(binary_flag(day, individual_prediction.loc["days_past"]))
    return np.concatenate((prediction, justification, flag), axis=None)

def call_questions(dataframe):
    correct_answer, possible_answers = correct_possible_answer(dataframe)
    prediction_stack = get_prediction_stack(dataframe)
    longest_day = prediction_stack["days_past"].max()

