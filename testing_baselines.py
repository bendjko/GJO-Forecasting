from baseline import *
import os

test_file = os.path.expanduser("~/Desktop/question_6.json")
dataframe = df(test_file)
prediction_stack = get_prediction_stack(dataframe)

day_forecast = daily_forecast(prediction_stack, 121)
day_past_forecast = past_forecast(prediction_stack, 121)
print(each_day(dataframe))
# print(dataframe)
#                                                                    0
# question_id                                                        6
# title              Will a trilateral meeting take place between C...
# possible_answers                                           [Yes, No]
# crowd_forecast                                            [0.0, 1.0]
# correct_answer                                                    No
# correct_forecast                                                 1.0
# preds              [[61, 2016-01-01T02:37:58Z, [0.0, 1.0], ], [63...
# open_date                                       2015-09-01T16:37:50Z
# close_date                                      2015-12-31T23:00:11Z
# question_duration                                                121

# print(prediction_stack)
#     user_id             date_time          pred                                               text  days_past  quartile
# 0        61  2016-01-01T02:37:58Z    [0.0, 1.0]                                                           121         4
# 1      6303  2015-12-31T23:27:32Z    [0.0, 1.0]  It is New Year's Eve and a holiday in Asia, so...        121         4
# 2      1131  2015-12-31T23:23:38Z    [0.0, 1.0]                                                           121         4
# 3       368  2015-12-31T21:47:51Z    [0.0, 1.0]                                                           121         4
# 4     14208  2015-12-31T20:55:55Z    [0.0, 1.0]                                                           121         4
# ..      ...                   ...           ...                                                ...        ...       ...
# 941      47  2015-09-02T00:37:31Z    [0.0, 1.0]                                                             0         0
# 942      34  2015-09-02T00:32:44Z    [0.1, 0.9]  With only four months left in 2015, conditions...          0         0
# 943      36  2015-09-02T00:29:28Z  [0.36, 0.64]  Good link by GJDrew .   As the Asia economic s...          0         0
# 944      26  2015-09-01T23:48:39Z  [0.15, 0.85]            Unlikely, these are very busy people ;)          0         0
# 945       2  2015-09-01T20:00:29Z  [0.35, 0.65]  Korea is looking for one.   http://www.bloombe...          0         0
# [946 rows x 6 columns]