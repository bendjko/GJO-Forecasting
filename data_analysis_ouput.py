from baseline import df
from data_analysis import *
import os

id_file = os.path.expanduser("~/Desktop/data_sub/")
save_path = os.path.expanduser("~/Desktop/data/")
data_file = os.path.expanduser("~/Desktop/question_1519.json")
sub_id_file = os.path.expanduser("~/Desktop/data_sub/11_questions.txt")
dataframe = df(data_file)
average_token_length_forecasts(dataframe)

# options:  2
# total token length: 0
# 1519
# total token length: 0
# 1222
# total token length: 0
# 1011
# total token length: 0
# 1020
# total token length: 0
# 981
# desired output:  [22.74423710208562, 62.28397583443677]

# options:  3
# desired output:  [18.485849056603772, 58.082844865583745]

# options:  4
# desired output:  [19.069620253164558, 66.89642661594341]

# options:  5
# desired output:  [19.75, 59.9666423760944]

# options:  6
# total token length: 0
# 2082
# desired output:  [16.17543859649123, 49.99206486582448]

# options:  7
# desired output:  [16.11904761904762, 57.183553605169244]

# options:  8
# desired output:  [13.2, 64.27549118978845]

# options:  9
# desired output:  [16.333333333333332, 50.07362378510041]

# options:  10
# desired output:  [17.25, 50.05732099475771]

# options:  11
# desired output:  [19.0, 58.34895833333333]

# options:  12
# desired output:  [14.8, 62.434271563624705]

# options:  13
# desired output:  [9.0, 20.72093023255814]