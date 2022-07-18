import pandas as pd
import json
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

def df(data_file):
  jdata = json.load(open(data_file))
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  close_date = datetime.fromisoformat(str(dataframe.loc["close_date", 0])[:-1])
  open_date = datetime.fromisoformat(str(dataframe.loc["open_date", 0])[:-1])
  question_duration = (close_date - open_date).days
  df2 = pd.DataFrame([question_duration], index=['question_duration'])
  dataframe = dataframe.append(df2)
  warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
  warnings.simplefilter(action='ignore', category=FutureWarning)
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
    i=3
    while (i<=3):
      if (days_past >= question_duration * i):
        quartile.append(i+1)
        break
      i-=1
  prediction_stack["days_past"] = day_past
  prediction_stack["quartile"] = quartile
  preds = prediction_stack.loc[:, "pred"]
  if len(prediction_stack.loc[0, "pred"]) == 1:
    for pred in preds:
      pred.append(1- pred[0])
  return prediction_stack

def which_questions(dataframe):
  no_of_answers = len(dataframe.loc["possible_answers", 0])
  return no_of_answers

def correct_possible_answer(dataframe):
  correct_answer = dataframe.loc['correct_answer', 0]
  possible_answers = dataframe.loc['possible_answers', 0]
  return correct_answer, possible_answers

def daily_forecast(prediction_stack, day):
  return prediction_stack[prediction_stack['days_past']==day]

def quartile_forecast(prediction_stack, quartile):
  return prediction_stack[prediction_stack['quartile']==quartile]

def active_forecast_ten_days_prior(prediction_stack, day):
 if day <= 10:
    first_filter = prediction_stack[prediction_stack['days_past']<=day].drop_duplicates(subset=['user_id'])
    second_filter = first_filter[first_filter['days_past'] == day]
    return second_filter
 else:
    first_filter= prediction_stack[prediction_stack['days_past']<=day]
    day2 = day - 10
    second_filter = first_filter[first_filter['days_past']>=day2].drop_duplicates(subset=['user_id'])
    third_filter = second_filter[second_filter['days_past'] == day]
    return third_filter

def correct_day_counter(correct_answer, possible_answers, prediction_index):
  predicted_answer = possible_answers[prediction_index]
  if correct_answer == predicted_answer:
    return 1
  else: 
    return 0

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

def baselines(dataframe):
  correct_answer, possible_answers = correct_possible_answer(dataframe)
  prediction_stack = get_prediction_stack(dataframe)
  longest_day = prediction_stack["days_past"].max()
  daily_forecast_majority_tracker = 0
  daily_forecast_weighted_tracker = 0
  active_forecast_majority_tracker=0
  active_forecast_weighted_tracker=0
  day_counter = 0
  while (longest_day >= 0):
    day_forecast = daily_forecast(prediction_stack, longest_day)
    if len(day_forecast['pred']) >= 1:
        day_past_forecast = active_forecast_ten_days_prior(prediction_stack, longest_day)
        daily_forecast_majority_tracker += correct_day_counter(correct_answer, possible_answers, majority_baseline(day_forecast))
        daily_forecast_weighted_tracker += correct_day_counter(correct_answer, possible_answers, weighted_baseline(day_forecast))
        active_forecast_majority_tracker += correct_day_counter(correct_answer, possible_answers, majority_baseline(day_past_forecast))
        active_forecast_weighted_tracker += correct_day_counter(correct_answer, possible_answers, weighted_baseline(day_past_forecast))
        day_counter+=1
    longest_day -=1
  return [daily_forecast_majority_tracker/day_counter, 
          daily_forecast_weighted_tracker/day_counter,
          active_forecast_majority_tracker/day_counter,
          active_forecast_weighted_tracker/day_counter
          ]

def all_questions_baseline(id_file, path):
  question_counter = 0
  total_baseline = np.array([0, 0, 0, 0])
  id_counter = 0
  with open(id_file, "r") as f:
    for line in f.readlines():
      data_file = f"{path}question_{int(line)}.json"
      id_counter +=1
      each_baseline = np.array(baselines(df(data_file)))
      total_baseline = np.add(total_baseline, each_baseline)
      question_counter +=1
  return total_baseline/question_counter

def export_questions_by_number_of_possible_answers(id_file, data_path, save_path):
  answer_count = 0
  with open(id_file, "r") as f:
    for line in f.readlines():
      question_id = int(line)  
      data_file = f"{data_path}question_{question_id}.json"
      answer_count = which_questions(df(data_file))
      sub_file = f"{save_path}{answer_count}_questions.txt"
      with open(sub_file, "a") as f:
        f.write(str(question_id))
        f.write("\n")

def which_forecasts(dataframe):
  prediction_stack = get_prediction_stack(dataframe)
  all_forecasts = prediction_stack
  justified_forecasts = prediction_stack[prediction_stack['text']!='']
  not_justified_forecasts = prediction_stack[prediction_stack['text']=='']
  return all_forecasts, justified_forecasts, not_justified_forecasts

def subset_forecast_count(sub_id_file, path):
  no_of_questions = np.array([0, 0, 0])
  with open(sub_id_file, "r") as f:
    for line in f.readlines():
      question_id = int(line)  
      data_file = f"{path}question_{question_id}.json"
      dataframe = df(data_file)
      all_forecasts, justified_forecasts, not_justified_forecasts = which_forecasts(dataframe)
      no_of_questions += [len(all_forecasts),len(justified_forecasts),len(not_justified_forecasts)]
  return no_of_questions

def loop_through_subset(id_file_path, data_file_path):
  options = 2
  while (options <= 13):
    id_file = f"{id_file_path}{options}_questions.txt"
    # subset_count = subset_forecast_count(id_file, data_file_path)
    baseline_sum = baseline_sum_by_question_no(id_file, data_file_path)
    print("options", options, baseline_sum)
    options+=1

def baseline_sum_by_question_no(sub_id_file, path):
  baseline_array = np.array([0, 0, 0, 0, 0, 0])
  with open(sub_id_file, "r") as f:
    for line in f.readlines():
      question_id = int(line)  
      data_file = f"{path}question_{question_id}.json"
      dataframe = df(data_file)
      arr =  baselines_for_subsets(dataframe)
      baseline_array += arr
  return baseline_array

def baselines_for_subsets(dataframe):
  correct_answer, possible_answers = correct_possible_answer(dataframe)
  all, text, no_text = which_forecasts(dataframe)
  mb_all = 0
  wb_all = 0
  mb_text = 0
  wb_text = 0
  mb_no = 0
  wb_no = 0
  if all.empty == False:
    mb_all += correct_day_counter(correct_answer, possible_answers, majority_baseline(all))
    wb_all += correct_day_counter(correct_answer, possible_answers, weighted_baseline(all))
    if no_text.empty == True:
     mb_text += correct_day_counter(correct_answer, possible_answers, majority_baseline(text))
     wb_text += correct_day_counter(correct_answer, possible_answers, weighted_baseline(text))
    elif text.empty == True:
     mb_no += correct_day_counter(correct_answer, possible_answers, majority_baseline(no_text))
     wb_no += correct_day_counter(correct_answer, possible_answers, weighted_baseline(no_text))
    else:
     mb_text += correct_day_counter(correct_answer, possible_answers, majority_baseline(text))
     wb_text += correct_day_counter(correct_answer, possible_answers, weighted_baseline(text))
     mb_no += correct_day_counter(correct_answer, possible_answers, majority_baseline(no_text))
     wb_no += correct_day_counter(correct_answer, possible_answers, weighted_baseline(no_text))
  return [mb_all, wb_all, mb_text, wb_text, mb_no, wb_no]

subset_id_file = os.path.expanduser("~/Desktop/data_sub/")
path = os.path.expanduser("~/Desktop/data/")
loop_through_subset(subset_id_file, path)

# id_file = os.path.expanduser("~/Desktop/id_file_clear.txt")
# print(all_questions_baseline(id_file, path))
# [daily forecast majority baseline, daily forecast weighed baseline, active forecast majority baseline, active forecast weighted baseline]
# [0.70786752 0.71518103 0.81939237 0.82636818]
# [0.71961468 0.72112447]

# [# of total forecasts, # of forecasts w/o justifications, # of forecasts w/ justifications]
# 2 [666582 164714 501868]
# 3 [148718  29192 119526]
# 4 [119398  25757  93641]
# 5 [220924  40803 180121]
# 6 [23764  3941 19823]
# 7 [25501  4964 20537]
# 8 [5432 1025 4407]
# 9 [3854  571 3283]
# 10 [3084  414 2670]
# 11 [2554  459 2095]
# 12 [8189 1765 6424]
# 13 [978  43 935]

# baseline output by no. of possible options
# [majority(all), weighted(all), majority(textual justification), weighted(text), majority(no textual justification), weighted(no text)]
# options 2 [792 796 788 794 789 797]
# options 3 [162 163 161 167 160 163]
# options 4 [109 116 114 119 109 116]
# options 5 [253 261 252 261 248 257]
# options 6 [27 33 30 31 27 34]
# options 7 [27 28 28 28 27 28]
# options 8 [7 7 7 7 7 7]
# options 9 [11 11 12  9 11 11]
# options 10 [5 5 4 5 5 5]
# options 11 [1 2 1 1 1 2]
# options 12 [4 4 4 4 4 4]
# options 13 [1 1 1 1 1 1]