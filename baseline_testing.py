import os
from baseline import *

def baseline_match_data(id_file, path):
  question_counter = 0
  forecast_counter = 0
  total_baseline = np.array([0.0, 0.0, 0.0, 0.0])
  with open(id_file, "r") as f:
    for line in f.readlines():
     if line != "\n":
      question_counter += 1
      data_file = f"{path}question_{int(line)}.json"
      print(question_counter,":", line)
      dataframe = df(data_file)
      justified_forecast = justified_forecasts(dataframe)
      forecast_counter += len(justified_forecast)
      total_baseline += np.array(justified_baselines(dataframe))
      question_counter_arr = np.array([question_counter, question_counter, question_counter, question_counter])
      print(total_baseline/question_counter_arr, forecast_counter)
  question_counter_arr = np.array([question_counter, question_counter, question_counter, question_counter])

  return total_baseline, question_counter, total_baseline/question_counter_arr, forecast_counter

def justified_forecasts(dataframe):
  prediction_stack = get_prediction_stack(dataframe)
  return prediction_stack[prediction_stack['text']!='']

def justified_baselines(dataframe):
  correct_answer, possible_answers = correct_possible_answer(dataframe)
  prediction_stack = justified_forecasts(dataframe)
  prediction_stack2 = get_prediction_stack(dataframe)
  longest_day = prediction_stack["days_past"].max()
#   longest_day_exp = prediction_stack2["days_past"].max()
  longest_day2 = longest_day
  
  mb_text = 0
  wb_text =0
  text_ct = 0
  mb_text2 =0
  wb_text2 =0
  text2_ct =0
  textmb = 0
  textwb = 0
  textmb2 = 0
  textwb2 = 0

  while (longest_day >= 0):
    day_forecast = daily_forecast(prediction_stack, longest_day)
    all, text, no_text = which_forecasts_pred(day_forecast)
    if all.empty == False:
      if text.empty == False:
        mb_text += correct_day_counter(correct_answer, possible_answers, majority_baseline(text))
        wb_text += correct_day_counter(correct_answer, possible_answers, weighted_baseline(text))
        text_ct += 1
    day_past_forecast = active_forecast_ten_days_prior(prediction_stack, longest_day)
    all2, text2, no_text2 = which_forecasts_pred(day_past_forecast)
    if all2.empty == False:
      if text2.empty == False:
        mb_text2 += correct_day_counter(correct_answer, possible_answers, majority_baseline(text2))
        wb_text2 += correct_day_counter(correct_answer, possible_answers, weighted_baseline(text2))
        text2_ct += 1
    longest_day -= 1
    

    textmb = mb_text/text_ct
    textwb = wb_text/text_ct
    textmb2 = mb_text2/text2_ct
    textwb2 = wb_text2/text2_ct
    # textmb = mb_text/longest_day2   
    # textwb = wb_text/longest_day2
    # textmb2 = mb_text2/longest_day2
    # textwb2 = wb_text2/longest_day2

  return [textmb, textwb, textmb2, textwb2]

id_file = os.path.expanduser("~/Desktop/saketh_id.txt")
path = os.path.expanduser("~/Desktop/data/")
print(baseline_match_data(id_file, path))

# 441 questions, 0.76015081, 0.77906721, 0.76038577, 0.7794903, 99247 forecasts
# array([0.76015081, 0.77906721, 0.76038577, 0.7794903 ]), 99247)
# 441, array([0.76015081, 0.77906721, 0.80156881, 0.81886236]), 99247)

# gotta fix active_forecast_ten_days_prior code