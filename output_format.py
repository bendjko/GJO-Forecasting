from baseline import *
import os

path = os.path.expanduser("~/Desktop/question_1951.json")
id_file = os.path.expanduser("~/Desktop/id_file.txt")
print(df(path))

# dataframe
#                                                                    0
# question_id                                                     1951
# title              Will the People's Republic of China's (PRC's) ...
# possible_answers                                           [Yes, No]
# crowd_forecast                                        [0.985, 0.015]
# correct_answer                                                   Yes
# correct_forecast                                               0.985
# preds              [[100715, 2021-12-31T15:07:35Z, [1.0], ], [103...
# open_date                                       2021-02-26T18:00:58Z
# close_date                                      2022-01-01T08:01:58Z
# question_duration                                                308
