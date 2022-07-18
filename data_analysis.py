from baseline import *
import re
import numpy as np
import pandas as pd
from pprint import pprint
import os
import gensim
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import spacy
import nltk
import ssl
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import logging
import warnings
from wordcloud import WordCloud
from nltk.tokenize import *

warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def token_length_questions(dataframe):
    question = dataframe.loc['title', 0]
    token_length = len(word_tokenize(question))
    return token_length

def average_token_length_forecasts(dataframe):
    all_forecasts, justified_forecasts, not_justified_forecasts = which_forecasts(dataframe)
    total_token_length = 0
    forecast_count = 0
    justified_forecasts = justified_forecasts['text']
    for text in justified_forecasts:
      token_length = len(word_tokenize(text))
      total_token_length += token_length
      forecast_count += 1
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

def get_titles(dataframe):
  return dataframe.loc['title', 0]

def title_from_data(question_id, path):
  data_file = f"{path}question_{question_id}.json"
  title = np.array(tokenize(get_titles(df(data_file))))
  return title

def tokenize(sentence):
  return gensim.utils.simple_preprocess(str(sentence), deacc=True)

def return_title(sub_id_path, options, path):
  df = np.array([""])
  sub_id_file = f"{sub_id_path}{options}_questions.txt"
  with open(sub_id_file, "r") as f:
    for line in f.readlines():
      question_id = int(line)  
      title = remove_stopwords(title_from_data(question_id, path))
      df = np.vstack([df, title])
    df = np.delete(df, 0, 0)
  return pd.DataFrame(df)

def remove_stopwords(df):
  stop_words = stopwords.words('english')
  return [word for word in df if word not in stop_words]

def lemmatization(df, allowed_postags):
    texts_out = []
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    for sent in df.itertuples(index=False):
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def build_lda(df):
  df = lemmatization(df, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
  dic = corpora.Dictionary(df)
  texts = df
  corpus = [dic.doc2bow(text) for text in texts]
  lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dic,
                                          #  num_topics=20, 
                                          #  random_state=100,
                                          #  update_every=1,
                                          #  chunksize=100,
                                          #  passes=10,
                                          #  alpha='auto',
                                           per_word_topics=True)
  for t in range(lda.num_topics):
    plt.figure()
    plt.imshow(WordCloud().fit_words(dict(lda.show_topic(t, 200))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()

# might not need at all
# def title_stack(sub_id_path, path):
#   df = np.array([""])
#   question_no = 2
#   while question_no <= 13:
#     titles = return_title(sub_id_path, question_no, path)
#     df = np.vstack([df, titles])
#     question_no += 1
#   df = np.delete(df, 0, 0)
#   return df

def loop_through_subset(id_file_path, data_file_path):
  options = 2
  while (options <= 13):
    id_file = f"{id_file_path}{options}_questions.txt"
    # print("options:", options)
    # print("avg. question length, avg. forecast length:", subset_question_forecast_length(id_file, data_file_path))
    df = return_title(id_file_path, options, data_file_path)
    build_lda(df)
    print("options:", options)
    options +=1

id = os.path.expanduser("~/Desktop/data_sub/")
data = os.path.expanduser("~/Desktop/data/")
test_data_file = os.path.expanduser("~/Desktop/question_1222.json")
# test_sub_id_file = os.path.expanduser("~/Desktop/data_sub/11_questions.txt")
dataframe = df(test_data_file)

# loop_through_subset(id_file, save_path)
# loop_through_subset(id_file, save_path)

print(return_title(id, 2, data))
# print(remove_stopwords(dataframe))
# print(lemmatization(dataframe, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))
# build_lda(dataframe)