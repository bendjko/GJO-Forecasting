from baseline import *
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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

class LSTMlayer(nn.Module):
    def __init__(self, input_size=258, hidden_size=256):
        super(LSTMlayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(self.lstm.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        output, _ = self.lstm(input)
        out = self.out(output)
        out = self.sigmoid(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
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

def binary_option_forecast_representation(day, individual_prediction):
    clf = Bert_Model()
    justification = clf(preprocess(individual_prediction["text"])).detach().numpy()
    prediction = int(binary_prediction_representation(individual_prediction["pred"]))
    flag = binary_flag(day, individual_prediction["days_past"])
    return np.concatenate((prediction, justification, flag), axis=None)

def question_representation(dataframe):
    clf = Bert_Model()
    place_holder = np.array([[0]])
    output = clf(preprocess(dataframe.loc["title"][0])).detach().numpy()
    return np.append(output, [0,0])

# concatenates question and justification into 3d tensor
# torch.Size([1, 2, 258])
def binary_option_input(question, forecast):
    input = np.vstack((question,forecast))
    input = torch.from_numpy(input)
    # input = torch.reshape(input, (258, 2))
    input = torch.unsqueeze(input, 0)
    return input

# concatenates each forecast into 3d tensor for final input to lstm
def return_input_tensor(preds):
    input_tensor = torch.empty(2, 258, 1)
    for ind in preds.index:
        pred = preds.iloc[ind]
        forecast_rep = binary_option_forecast_representation(1,pred)
    row_exclude = 1
    input_tensor = torch.from_numpy(input_tensor)
    input_tensor = torch.cat((input_tensor[:row_exclude],input_tensor[row_exclude+1:]))
    return input_tensor


def call_questions(dataframe):
    correct_answer, possible_answers = correct_possible_answer(dataframe)
    prediction_stack = get_prediction_stack(dataframe)
    longest_day = prediction_stack["days_past"].max()


