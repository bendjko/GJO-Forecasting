from asyncore import loop
from data_analysis import *
from baseline import *
import os

id_file = os.path.expanduser("~/Desktop/data_sub/")
save_path = os.path.expanduser("~/Desktop/data/")
data_file = os.path.expanduser("~/Desktop/question_1951.json")
sub_id_file = os.path.expanduser("~/Desktop/data_sub/11_questions.txt")
loop_through_subset(id_file, save_path)