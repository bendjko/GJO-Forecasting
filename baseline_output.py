from baseline import *
import os

id_file = os.path.expanduser("~/Desktop/id_file_clear.txt")
subset_id_file = os.path.expanduser("~/Desktop/data_sub/")
path = os.path.expanduser("~/Desktop/data/")

print(all_questions_baseline(id_file, path))
# [daily forecast majority baseline, daily forecast weighed baseline, active forecast majority baseline, active forecast weighted baseline]
# [0.70786752 0.71518103 0.81939237 0.82636818]

loop_through_subset(subset_id_file, path)
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