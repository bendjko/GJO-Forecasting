from tkinter import W
import pandas as pd
import os
import json
import numpy as np
import datetime 
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

def df(data_file):
  # data_file = f"question_{question_id}.json"
  jdata = json.load(open(data_file))
  dataframe['question_duration'] = (dataframe['close_date'] - dataframe['open_date']).dt.days
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  return dataframe

def get_prediction_stack(dataframe):
  prediction_stack = []
  for prediction in dataframe.loc["preds", :]:
    prediction_stack = np.row_stack(prediction)
  prediction_stack = pd.DataFrame(prediction_stack)
  prediction_stack = prediction_stack.rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})
  prediction_stack["days_past"] = (prediction_stack["date_time"] - dataframe['open_date']).dt.days
  return prediction_stack

# consider where forecast options are binary, thus not giving the second option to be considered for forecasting answer
def majority_baseline(dataframe):
  # find max by row
  dummyclf = DummyClassifier(strategy="most_frequent")
  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  if len(dataframe.loc["possible_answers", :]) == 2:
    for pred in preds:
      if 
  else:
    fit_set = preds
    max_index = []
    for pred in fit_set:
      max_index_row = np.argmax(pred, axis=1)
      max_index.append(max_index_row)
    dummyclf.fit(preds, fit_set)
    dummyclf.predict(preds)

def weighted_baseline(dataframe):
  # find max by column
  dummyclf = DummyClassifier(strategy="stratified")
  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  prob_sum = np.sum(preds, axis=1)

  dummyclf.predict(prob_sum)
  






