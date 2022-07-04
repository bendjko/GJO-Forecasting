from tkinter import W
import pandas as pd
import os
import json
import numpy as np
import datetime as dt
from datetime import date, datetime, timedelta
from responses import target
from sklearn.dummy import DummyRegressor

def df(data_file):
  jdata = json.load(open(data_file))
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  close_date = datetime.fromisoformat(str(dataframe.loc["close_date", 0])[:-1])
  open_date = datetime.fromisoformat(str(dataframe.loc["open_date", 0])[:-1])
  question_duration = (close_date - open_date).days
  df2 = pd.DataFrame([question_duration], index=['question_duration'])
  dataframe = dataframe.append(df2)
  return dataframe

def get_prediction_stack(dataframe):
  prediction_stack = []
  for prediction in dataframe.loc["preds", :]:
    prediction_stack = np.row_stack(prediction)

  prediction_stack = pd.DataFrame(prediction_stack)
  prediction_stack = prediction_stack.rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})
  # prediction_stack["days_past"] = (prediction_stack["date_time"] - dataframe['open_date']).dt.days
  answer_date = datetime.fromisoformat(str(prediction_stack.loc["date_time"])[:-1])
  open_date = datetime.fromisoformat(str(dataframe.loc["open_date", 0])[:-1])
  question_duration = (answer_date - open_date).days
  df2 = pd.DataFrame([question_duration], index=['question_duration'])
  dataframe = dataframe.append(df2)
  print(dataframe)
  return prediction_stack

def which_questions(dataframe):
  no_of_answers = len(dataframe.loc["possible_answers", :])
  return no_of_answers

# deal with all daily and all pasts since day would not be our parameter
def all_daily_forecast(prediction_stack, day):
  return prediction_stack[prediction_stack['days_past']==day]

def all_pasts_forecast(predicition_stack, day):
  return predicition_stack[predicition_stack['days_past']<=day]

def latest_forecast(dataframe):
  prediction_stack = get_prediction_stack(dataframe)
  latest_predictions = prediction_stack.drop_duplicates(subset=['user_id'])
  return latest_predictions

def which_forecasts(dataframe):
  prediction_stack = get_prediction_stack(dataframe)
  all_forecasts = prediction_stack
  justified_forecasts = prediction_stack[prediction_stack.loc['text']!='']
  return all_forecasts, justified_forecasts

def get_correct_forecast_in_normalized_array(dataframe):
  possible_answers = dataframe.loc['possible_answers', 0]
  correct_answer = dataframe.loc['correct_answer', 0]
  correct_index = possible_answers.index(correct_answer)
  normalized_array = possible_answers
  for i in range(len(normalized_array)):
    if i == correct_index:
      normalized_array[i] = 1
    else:
      normalized_array[i]=0
  return normalized_array
  
def majority_baseline(dataframe):
  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  if len(dataframe.loc["possible_answers", 0]) == 2:
    for pred in preds:
      pred.append(1- pred[0])
  total_prob = np.zeros_like(preds[0])
  for pred in preds:
    pred = pd.DataFrame(pred).transpose()
    new_pred = np.zeros_like(pred.values)
    new_pred[np.arange(len(pred)), pred.values.argmax(1)] = 1
    total_prob += np.sum(new_pred, axis = 0)
  prediction_index = total_prob.argmax()
  return dataframe.loc['possible_answers', 0][prediction_index]

def weighted_baseline(dataframe):
  preds = get_prediction_stack(dataframe).loc[:, "pred"]
  if len(dataframe.loc["possible_answers", 0]) == 2:
    for pred in preds:
      pred.append(1- pred[0])
  total_prob = np.zeros_like(preds[0])
  for pred in preds:
    pred = pd.DataFrame(pred).transpose()
    total_prob += np.sum(pred, axis = 0)
  prediction_index = total_prob.argmax()
  return dataframe.loc['possible_answers', 0][prediction_index]

def variations(data_file):
  return None

test_file = os.path.expanduser("~/Desktop/question_6.json")
dataframe = df(test_file)
print(get_prediction_stack(dataframe))
# print(latest_forecast(dataframe))