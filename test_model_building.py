from model_building import *
from baseline import *
import os

test_file = os.path.expanduser("~/Desktop/question_1951.json")
dataframe = df(test_file)
question_rep = question_representation(dataframe)
preds = get_prediction_stack(dataframe)

test_for = preds.loc[437]
test_forecast = binary_option_forecast_representation(2, test_for)
test_input = binary_option_input(question_rep, test_forecast)
# print(test_input.size())

# lstmlayer = LSTMlayer()
# print(lstmlayer(test_input))

input_dim = 258
hidden_dim = 256
layer_dim = 1
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

print(model(test_input))

# outline
# concatenate input forecast sequence into 3d tensor
# remove first tensor (randomized tensor) and reconfirm the size 
# using .view from torch, feed a sequence to lstm layer
# 
