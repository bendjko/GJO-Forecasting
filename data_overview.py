import pandas as pd
import os
import json
import numpy as np

def df(data_file):
  jdata = json.load(open(data_file))
  dataframe = pd.DataFrame.from_dict(jdata, orient='index')
  return dataframe

def get_prediction_stack(data_file):
  dataframe = df(data_file)
  prediction_stack = []
  for prediction in dataframe.loc["preds", :]:
    prediction_stack = np.vstack(prediction)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 
  prediction_stack = pd.DataFrame(prediction_stack)
  prediction_stack = prediction_stack.rename(columns={0: "user_id", 1: "date_time", 2:"pred", 3:"text"})
  return prediction_stack
  
def basic_stats(data_file):
  table = get_prediction_stack(data_file)
  forecast_count = len(table.index)
  written_justification_count = 0
  for text_input in table.loc[:, "text"]:
    if text_input != '':
      written_justification_count += 1
  return forecast_count, written_justification_count

def loop_all_data(id_file, path):
  total_forecast_count = 0
  total_written_justification_count = 0
  with open(id_file, "r") as f:
    for line in f.readlines():
      data_file = f"{path}question_{int(line)}.json"
      forecast_count, written_justification_count = basic_stats(data_file)
      total_forecast_count += forecast_count
      total_written_justification_count += written_justification_count
  return total_forecast_count, total_written_justification_count

def print_data_table(id_file, path):
  total_forecast_count, total_written_justification_count = loop_all_data(id_file, path)  
  dd = {'Question': 1788, 'Forecast': total_forecast_count, 'Written Justification': total_written_justification_count}
  df = pd.DataFrame(data=dd, index=['Count'])
  print(df)

# path = os.path.expanduser("~/Desktop/data/")
# id_file = os.path.expanduser("~/Desktop/id_file_2.txt")

# print_data_table(id_file, path)

path = os.path.expanduser("~/Desktop/data/")
id_file = os.path.expanduser("~/Desktop/id_file_2.txt")

print_data_table(id_file, path)