from baseline import *
import os

def export_questions_by_possible_answers(id_file, data_path, save_path):
  answer_count = 0
  id_counter = 0
  with open(id_file, "r") as f:
    for line in f.readlines():
      question_id = int(line)  
      data_file = f"{data_path}question_{question_id}.json"
      id_counter +=1
      print(id_counter)
      answer_count = which_questions(df(data_file))
      sub_file = f"{save_path}{answer_count}_questions.txt"
      with open(sub_file, "a") as f:
        f.write(str(question_id))
        f.write("\n")


id_file = os.path.expanduser("~/Desktop/id_file_clear.txt")
data_path = os.path.expanduser("~/Desktop/data/")
save_path = os.path.expanduser("~/Desktop/data_sub/")
export_questions_by_possible_answers(id_file, data_path, save_path)
      
      
 