from tkinter import W
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
from responses import target

def read_id_file(id_file, path):
  with open(id_file, "r") as f:
    for line in f.readlines():
      dataframe = df(f"{path}question_{int(line)}.json")
      prediction_stack = get_prediction_stack(dataframe)

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
  prediction_stack = pd.DataFrame(prediction_stack).rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})
  open_date = datetime.fromisoformat(str(dataframe.loc["open_date", 0])[:-1])
  day_past = []
  quartile = []
  question_duration = dataframe.loc["question_duration", 0]/4
  for answer_date in prediction_stack.loc[:, "date_time"]:
    days_past = (datetime.fromisoformat(str(answer_date[:-1])) - open_date).days
    day_past.append(days_past)
    i=4
    while (i<=4):
      if (days_past >= question_duration * i):
        quartile.append(i)
        break
      i-=1
  prediction_stack["days_past"] = day_past
  prediction_stack["quartile"] = quartile
  preds = prediction_stack.loc[:, "pred"]
  if len(prediction_stack.loc[0, "pred"]) == 1:
    for pred in preds:
      pred.append(1- pred[0])
  return prediction_stack

# last thing to deal with since it's not returning prediction_stacks
def which_questions(dataframe):
  no_of_answers = len(dataframe.loc["possible_answers", :])
  return no_of_answers

def correct_possible_answer(dataframe):
  correct_answer = dataframe.loc['correct_answer', 0]
  possible_answers = dataframe.loc['possible_answers', 0]
  return correct_answer, possible_answers

def daily_forecast(prediction_stack, day):
  return prediction_stack[prediction_stack['days_past']==day]

def past_forecast(prediction_stack, day):
  return prediction_stack[prediction_stack['days_past']<=day]

def last_forecast(prediction_stack):
  latest_predictions = prediction_stack.drop_duplicates(subset=['user_id'])
  return latest_predictions

def each_day(dataframe):
  correct_answer, possible_answers = correct_possible_answer(dataframe)
  prediction_stack = get_prediction_stack(dataframe)
  longest_day = prediction_stack["days_past"].max()
  daily_forecast_majority_tracker = 0
  daily_forecast_weighted_tracker = 0
  past_forecast_majority_tracker = 0
  past_forecast_weighted_tracker = 0
  day_counter = 0
  while (longest_day >= 0):
    day_forecast = daily_forecast(prediction_stack, longest_day)
    if len(day_forecast['pred']) >= 1:
        day_past_forecast = past_forecast(prediction_stack, longest_day)
        daily_forecast_majority_tracker += correct_day_counter(correct_answer, possible_answers, majority_baseline(day_forecast))
        daily_forecast_weighted_tracker += correct_day_counter(correct_answer, possible_answers, weighted_baseline(day_forecast))
        past_forecast_majority_tracker += correct_day_counter(correct_answer, possible_answers, majority_baseline(day_past_forecast))
        past_forecast_weighted_tracker += correct_day_counter(correct_answer, possible_answers, weighted_baseline(day_past_forecast))
        day_counter+=1
    longest_day -=1
  return [daily_forecast_majority_tracker/day_counter, daily_forecast_weighted_tracker/day_counter, past_forecast_majority_tracker/day_counter, past_forecast_weighted_tracker/day_counter]

def correct_day_counter(correct_answer, possible_answers, prediction_index):
  predicted_answer = possible_answers[prediction_index]
  if correct_answer == predicted_answer:
    return 1
  else: 
    return 0

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

# baselines
def majority_baseline(prediction_stack):
  preds = prediction_stack.loc[:, "pred"]
  total_prob = np.zeros_like(preds.iloc[0])
  for pred in preds:
    pred = pd.DataFrame(pred).transpose()
    new_pred = np.zeros_like(pred.values)
    new_pred[np.arange(len(pred)), pred.values.argmax(1)] = 1
    total_prob += np.sum(new_pred, axis = 0)
  prediction_index = total_prob.argmax()
  return prediction_index

def weighted_baseline(prediction_stack):
  preds = prediction_stack.loc[:, "pred"]
  total_prob = np.zeros_like(preds.iloc[0])
  for pred in preds:
    pred = pd.DataFrame(pred).transpose()
    total_prob += np.sum(pred, axis = 0)
  prediction_index = total_prob.argmax()
  return prediction_index

# def accuracy(prediction_stack):
#   length = len(prediction_stack)
#   while(length >= 0):