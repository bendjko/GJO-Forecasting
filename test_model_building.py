from model_building import *
from baseline import *
import os


binary_questions = os.path.expanduser("~/Dekstop/data_sub/2_questions.txt")

test_file = os.path.expanduser("~/Desktop/question_1951.json")
dataframe = df(test_file)
preds = get_prediction_stack(dataframe)
clf = Bert_Model()
for ind in preds.index:
    forecast_input = str(preds['text'][ind])
    print(clf(preprocess_forecast(forecast_input)).size())
    # pred_input = preds['pred'][ind]
    # print(binary_prediction_representation(pred_input))



