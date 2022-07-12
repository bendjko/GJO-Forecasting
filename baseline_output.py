from baseline import *
import os

id_file = os.path.expanduser("~/Desktop/id_file_clear.txt")
save_path = os.path.expanduser("~/Desktop/data/")

print(all_questions_baseline(id_file, save_path))