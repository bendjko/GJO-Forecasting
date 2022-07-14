from data_analysis import *
from baseline import *
import os

id_file = os.path.expanduser("~/Desktop/data_sub/")
save_path = os.path.expanduser("~/Desktop/data/")

loop_through_subset(id_file, save_path)
