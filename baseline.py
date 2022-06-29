from tkinter import W
import pandas as pd
import os
import json
import numpy as np
import datetime 
from sklearn.dummy import DummyRegressor

def df(data_file):
  jdata = json.load(open(data_file))
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  dataframe['question_duration'] = (dataframe['close_date'] - dataframe['open_date']).dt.days
  return dataframe

def get_prediction_stack(dataframe):
  prediction_stack = []
  for prediction in dataframe.loc["preds", :]:
    prediction_stack = np.row_stack(prediction)
  prediction_stack = pd.DataFrame(prediction_stack)
  prediction_stack = prediction_stack.rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})
  prediction_stack["days_past"] = (prediction_stack["date_time"] - dataframe['open_date']).dt.days
  return prediction_stack

def which_questions(dataframe):
  no_of_answers = len(dataframe.loc["possible_answers", :])
  return no_of_answers

def which_forecasters(dataframe):
  days_open = dataframe['question_duration']
  prediction_stack = get_prediction_stack(dataframe)
  latest_predictions = prediction_stack.drop_duplicates(subset=['user_id'])
  return latest_predictions

# deal with all daily and all pasts since day would not be our parameter
def all_daily_forecast(prediction_stack, day):
  return prediction_stack[prediction_stack['days_past']==day]

def all_pasts_forecast(predicition_stack, day):
  return predicition_stack[predicition_stack['days_past']<=day]

def which_forecasts(dataframe):
  prediction_stack = get_prediction_stack(dataframe)
  all_forecasts = prediction_stack
  justified_forecasts = prediction_stack[prediction_stack.loc['text']!='']
  return all_forecasts, justified_forecasts

def majority_baseline(dataframe):
    dummyclf = Du(strategy="most_frequent")
    preds = get_prediction_stack(dataframe).loc[:, "pred"]

# consider where forecast options are binary, thus not giving the second option to be considered for forecasting answer
def majority_baselines(dataframe):
  # find max by row
  dummyclf = DummyClassifier(strategy="most_frequent")
  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  # test this if statement using question 6 - binary option questions
  if len(dataframe.loc["possible_answers", :]) == 2:
    for pred in preds:
      pred.append(1- pred[0])
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
  