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

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

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
    return np.concatenate((prediction, justification, flag), axis=None)

def question_representation(dataframe):
    clf = Bert_Model()
    return clf(preprocess(dataframe.loc["title"]))

def call_questions(dataframe):
    correct_answer, possible_answers = correct_possible_answer(dataframe)
    prediction_stack = get_prediction_stack(dataframe)
    longest_day = prediction_stack["days_past"].max()

test_file = os.path.expanduser("~/Desktop/question_1951.json")
dataframe = df(test_file)
preds = get_prediction_stack(dataframe)
clf = Bert_Model()
for ind in preds.index:
    pred = preds.iloc[ind]
    print(binary_option_forecast_representation(1,pred))


