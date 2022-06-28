from tkinter import W
import pandas as pd
import os
import json
import numpy as np
import datetime 
from sklearn.dummy import DummyClassifier

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

def majority_baseline(dataframe):
    dummyclf = DummyClassifier(strategy="most_frequent")
    preds = get_prediction_stack(dataframe).loc[:, "pred"]
    


def baseline_data(dataframe):

# Criterion 3 - which forecasts?
# # preds, boolean on textual input
# - all
# - only with textual justifications
    preds = get_prediction_stack(dataframe)
    yes_text_preds = []
    no_text_preds = []
    for text_input in pd.DataFrame(preds.loc[:, "text"]):
        if text_input != None:
            yes_text_preds.append(text_input)
        else:
            no_text_preds.append(text_input)
# Criterion 2 - which forecasters?
# # preds, user_id
# - all daily forecasts
# 
    days = preds.loc[:, "days_past"]
    day = 0
    daystack = []
    question_lifetime = dataframe["question_duration"]
    while (day<=question_lifetime):
        for pred in preds:
            df.loc[df['column_name']]
            daystack[day] = np.vstack(pred)
        day+=1


# - all pasts forecasts (possible with repeats from the same forecaster)
# - last forecast by each forecaster who has participated (current day or past, but only the last one per forecaster)

def which_questions(dataframe):
# Criterion 1 - which questions?
# # possible_answers
# - all
# - with 2 answers
# - with 3 answers
# - with 4 answers
# - and so on
    no_of_answers = len(dataframe["possible_answers"])






