from tkinter import W
import pandas as pd
import os
import json
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from sklearn.dummy import DummyRegressor

def df(data_file):
  jdata = json.load(open(data_file))
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  # dataframe['question_duration'] = (dataframe["close_date"] - dataframe["open_date"]).dt.days
  return dataframe

def get_prediction_stack(dataframe):
  prediction_stack = []
  for prediction in dataframe.loc["preds", :]:
    prediction_stack = np.row_stack(prediction)
  prediction_stack = pd.DataFrame(prediction_stack)
  prediction_stack = prediction_stack.rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})
  # prediction_stack["days_past"] = (prediction_stack["date_time"] - dataframe['open_date']).dt.days
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
  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  if len(dataframe.loc["possible_answers", :]) == 2:
    for pred in preds:
      pred.append(1- pred[0])
  print (preds)
  majority = preds
  for pred in majority:
    zero_and_one = np.zeros_like(pred)
    zero_and_one[np.arange(len(pred)), pred.argmax(1)] = 1

  total_prob = np.sum(majority, axis=0)
  scaled_prob = total_prob / total_prob.sum()

  zero_and_one = np.zeros_like(total_prob)
  zero_and_one[np.arange(len(total_prob)), total_prob.argmax(1)] = 1
  
  dummy_regr = DummyRegressor(strategy="mean")
  #test purpose
  print(total_prob, scaled_prob, zero_and_one)

  dummy_regr.fit(scaled_prob, zero_and_one)
  dummy_regr.predict(scaled_prob)
  dummy_regr.score(scaled_prob, zero_and_one)

def weighted_baseline(dataframe):
  # find max by column

  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  total_prob = np.sum(preds, axis=0)
  scaled_prob = total_prob / total_prob.sum()

  zero_and_one = np.zeros_like(total_prob)
  zero_and_one[np.arange(len(total_prob)), total_prob.argmax(1)] = 1
  
  dummy_regr = DummyRegressor(strategy="mean")
  #test purpose
  print(total_prob, scaled_prob, zero_and_one)

  dummy_regr.fit(scaled_prob, zero_and_one)
  dummy_regr.predict(scaled_prob)
  dummy_regr.score(scaled_prob, zero_and_one)

test_data_file = os.path.expanduser("~/Desktop/question_6.json")
dataframe = df(test_data_file)
majority_baseline(dataframe)

  