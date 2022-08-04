from re import X
from model_building import *
from baseline import *
import os
from transformers import BertConfig

test_file = os.path.expanduser("~/Desktop/question_1951.json")
dataframe = df(test_file)
preds = get_prediction_stack(dataframe)
# input = preprocess_forecast(preds['text'][0])
clf = Bert_Model()
for ind in preds.index:
    input = str(preds['text'][ind])
    print(clf(preprocess_forecast(input)))
# .last_hidden_state.shape
# config = BertConfig.from_pretrained("bert-base-uncased")
# print(config.hidden_size)


