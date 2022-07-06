from baseline import *
import os

def all_days(dataframe):
    prediction_stack = get_prediction_stack(dataframe)
    question_duration = dataframe.loc['question_duration', 0]
    i = question_duration
    # while (i<= question_duration):
    return None

# accuracy = average percentage of days a model calls a question correctly
# i have to write a method so that it returns prediction stack for each day - daily forecast
# daily_forecast cannot iterate through days
def accuracy_for_all_lifetime(dataframe):
    prediction_stack = get_prediction_stack(dataframe)
    total_preds = len(prediction_stack)
    # the correct_answer would be a problem 
    # since dataframe parameter is not 
    longest_days = prediction_stack.loc["days_past"].max()
    correct_answer = dataframe.loc['correct_answer', 0]
    majority_baseline_output = majority_baseline(dataframe)
    weighted_baseline_output = weighted_baseline_output(dataframe)
    # correct_answer = 0
    # total_answer = 0
    return 



