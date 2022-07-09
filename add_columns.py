from baseline import *
import os
import pandas as pd
import numpy as np

def export_questions_by_possible_answers(id_file, data_path, save_path):
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

# id_file = os.path.expanduser("~/Desktop/id_file_clear.txt")
# data_path = os.path.expanduser("~/Desktop/data/")
# save_path = os.path.expanduser("~/Desktop/data_sub/")
# export_questions_by_possible_answers(id_file, data_path, save_path)

def updated_active_forecasts(prediction_stack, day):
  if day <= 10:
    first_filter = prediction_stack[prediction_stack['days_past']<=day].drop_duplicates(subset=['user_id'])

    second_filter = first_filter[first_filter['days_past']==day]

    return second_filter
  else: 
    day-=10
    first_filter= prediction_stack[prediction_stack['days_past']>=day].drop_duplicates(subset=['user_id'])

    second_filter = first_filter[first_filter['days_past']==day]

    return second_filter

def fixed_baselines(dataframe):
  correct_answer, possible_answers = correct_possible_answer(dataframe)
  prediction_stack = get_prediction_stack(dataframe)
  longest_day = prediction_stack["days_past"].max()
  active_forecast_majority_tracker=0
  active_forecast_weighted_tracker=0
  day_counter = 0
  while (longest_day >= 0):
    day_past_forecast = updated_active_forecasts(prediction_stack, longest_day)
    if len(day_past_forecast['pred']) >= 1:
        active_forecast_majority_tracker += correct_day_counter(correct_answer, possible_answers, majority_baseline(day_past_forecast))
        active_forecast_weighted_tracker += correct_day_counter(correct_answer, possible_answers, weighted_baseline(day_past_forecast))
        day_counter+=1
    longest_day -=1
  if day_counter > 0:
    return [active_forecast_majority_tracker/day_counter, active_forecast_weighted_tracker/day_counter]

def new_all_questions_baseline(id_file, path):
  question_counter = 0
  total_baseline = np.array([0, 0])
  id_counter = 0
  with open(id_file, "r") as f:
    for line in f.readlines():
      data_file = f"{path}question_{int(line)}.json"
      id_counter +=1
      print(id_counter)
      data = 
      each_baseline = np.array(fixed_baselines(df(data_file)))
      total_baseline = np.add(total_baseline, each_baseline)
      question_counter +=1
  return total_baseline/question_counter

id_file = os.path.expanduser("~/Desktop/id_file_clear.txt")
save_path = os.path.expanduser("~/Desktop/data/")

print(new_all_questions_baseline(id_file, save_path))

def subset_baseline(sub_id_file, path):

    return None

def subset_textual_baseline(sub_id_file, path):
    return None