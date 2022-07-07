from baseline import *
import os

test_file = os.path.expanduser("~/Desktop/question_6.json")
test_file_1 = os.path.expanduser("~/Desktop/question_2348.json")
test_file_2 = os.path.expanduser("~/Desktop/question_2329.json")
test_file_3 = os.path.expanduser("~/Desktop/question_2323.json")

id_file = os.path.expanduser("~/Desktop/test_id.txt")
save_path = os.path.expanduser("~/Desktop/")

print(baselines(df(test_file)))
print(baselines(df(test_file_1)))
print(baselines(df(test_file_2)))
print(baselines(df(test_file_3)))
print(all_questions_baseline(id_file, save_path))