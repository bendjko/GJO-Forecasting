from model_building import *
from baseline import *
import os


binary_questions = os.path.expanduser("~/Dekstop/data_sub/2_questions.txt")

test_file = os.path.expanduser("~/Desktop/question_1951.json")
dataframe = df(test_file)
preds = get_prediction_stack(dataframe)
clf = Bert_Model()
for ind in preds.index:
    # justification = preprocess(preds.iloc[ind]["text"])
    # print(justification)

    pred = preds.iloc[ind]
    print(pred)
    print(binary_option_forecast_representation(1, pred))
    # size = 1D 256

    # pred_input = preds['pred'][ind]
    # print(binary_prediction_representation(pred_input))



