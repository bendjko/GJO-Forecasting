from baseline import *
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

import nltk
import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.tokenize import *

# Based on number of possible options:
# compare length of the question 
# compare length of the textual justifications
# compare # of textual justifications
def token_length_questions(dataframe):
    question = dataframe.loc['title', 0]
    # nom = gensim.utils.simple_preprocess(str(question), deacc=True)
    # print(len(nom))
    token_length = len(word_tokenize(question))
    # print(word_tokenize(question))
    return token_length

def average_token_length_forecasts(dataframe):
    all_forecasts, justified_forecasts, not_justified_forecasts = which_forecasts(dataframe)
    total_token_length = 0
    forecast_count = 0
    justified_forecasts = justified_forecasts['text']
    for text in justified_forecasts:
      token_length = len(word_tokenize(text))
      if token_length > 0:
        total_token_length += token_length
        forecast_count += 1
      else:
        print(text)
    if forecast_count > 0:
      return total_token_length/forecast_count
    else: 
      return 0

def subset_question_forecast_length(sub_id_file, path):
  total_question_length = 0
  total_forecast_length = 0
  id_count = 0
  with open(sub_id_file, "r") as f:
    for line in f.readlines():
      question_id = int(line)  
      data_file = f"{path}question_{question_id}.json"
      dataframe = df(data_file)
      total_forecast_length += average_token_length_forecasts(dataframe)
      total_question_length += token_length_questions(dataframe)
      id_count += 1
  return [total_question_length/id_count, total_forecast_length/id_count]

# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# def lda_topic_visualization(dataframe):
#     question = sent_to_words(dataframe.loc['title', 0])
#     return None

def loop_through_subset(id_file_path, data_file_path):
  options = 2
  while (options <= 13):
    id_file = f"{id_file_path}{options}_questions.txt"
    print("options: ", options)
    print("desired output: ", subset_question_forecast_length(id_file, data_file_path))
    options +=1
    # yield subset_question_forecast_length(id_file, data_file_path)

# problem
# justified forecasts not being returned correctly
# why is this showing three output in array????
