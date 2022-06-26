import pandas as pd
import os
import json
import numpy as np

def df(data_file):
  # data_file = f"question_{question_id}.json"
  jdata = json.load(open(data_file))
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  predictions = dataframe.loc["preds", :]
  prediction_stack = []
  for prediction in predictions:
    prediction_stack = np.row_stack(prediction)
  prediction_stack = pd.DataFrame(prediction_stack)
  prediction_stack = prediction_stack.rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})



test_path = os.path.expanduser("~/Desktop/question_2418.json")
df(test_path)