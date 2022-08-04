from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

binary_questions = os.path.expanduser("~/Dekstop/data_sub/2_questions.txt")

# class Bert_Model(nn.Module):
#    def __init__(self):
#        super(Bert_Model, self).__init__()
#        self.bert = BertModel.from_pretrained('bert-base-uncased')
#        self.drop = nn.Dropout(0.5)
#        self.out = nn.Linear(768, 2)
#        self.act = nn.ReLU()
#    def forward(self, input):
#        output = self.bert(input)
#        output = self.act(self.drop(self.out(output)))
#        return output

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
       return out

def preprocess_forecast(text):
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

def prediction_representation()