from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
import re
import time
import json 
import os

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
driver.get("https://www.gjopen.com")
username = driver.find_element(By.XPATH, "//*[@id='user_email']")
password = driver.find_element(By.XPATH, "//*[@id='user_password']")

# username.send_keys("your_email@email.com")
# password.send_keys("your_password")

driver.find_element(By.XPATH, "//*[@id='new_user']/input[2]").click()

def auto_scroll(question_id):
  link_template = f"https://www.gjopen.com/questions/{question_id}"
  driver.get(link_template)
  time.sleep(3) 
  SCROLL_PAUSE_TIME = 2
  last_height = driver.execute_script("return document.body.scrollHeight")
  while True:
      driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
      time.sleep(SCROLL_PAUSE_TIME)
      new_height = driver.execute_script("return document.body.scrollHeight")
      if new_height == last_height:
          break
      last_height = new_height
  soup = BeautifulSoup(driver.page_source, "html.parser")
  return soup

def get_question_page(question_id):
  r = requests.get(f"https://www.gjopen.com/questions/{question_id}")
  soup = BeautifulSoup(r.content, 'html.parser')

  title = soup.find(id="question-name-header").find("p").text

  correct_circle = soup.find("i", class_="fa fa-check-circle")
  correct_answer = correct_circle.find_parent("td").find_previous_sibling("td").text

  possible_answers = []
  for answer in soup.find("table", class_ = "table table-striped consensus-table").find_all("td", class_="text-center"):
    possible_answer = answer.find_previous_sibling().text
    possible_answers.append(possible_answer)
  
  correct_forecast = float(correct_circle.find_parent("td", class_= "text-center").find_next_sibling("td", class_="text-right").text[:-1])/100

  crowd_forecast = []
  for forecast in soup.find(class_ = "table table-striped consensus-table").find_all("td", class_="text-right"):
    forecast = float(forecast.text.replace("%",""))/100
    crowd_forecast.append(forecast)

  date = soup.find("div", class_="question-openclose").find_all("small")
  open_date = date[0].find("span").get("data-localizable-timestamp")
  close_date = date[1].find("span").get("data-localizable-timestamp")

  return title, possible_answers, crowd_forecast, correct_answer, correct_forecast, open_date, close_date

def get_forecasts(question_id):
  page = auto_scroll(question_id)
  preds = []
  for comment in page.find_all("div", class_="flyover-comment"):
    if "flyover-comment-load-more" not in comment.get("class") and "flyover-comment-reply" not in comment.get("class"):
      comment_pred = comment.find("div", class_="prediction-set")
      text_input = ""
      if comment_pred:
        bodies = comment.find("div", class_="flyover-comment-content").find_all("p")
        if bodies != None:
          for body in bodies:
            body = body.getText()
            text_input += body
            text_input += " "
        text_input = text_input[:-1]
        date = comment.find("div", class_="flyover-comment-date").find("span").get("data-localizable-timestamp")
        user_id = comment_pred.find("a", class_="membership-link").get("href")
        user_id = int(re.match(r"https:\/\/www\.gjopen\.com\/memberships\/(\d+)", user_id).groups()[0])
        pred = []
        for row in comment_pred.find_all(class_="row row-condensed"):
          prob = float(row.find("div", class_="probability-col").text.replace("%",""))/100
          pred.append((prob))
        preds.append((user_id, date, pred, text_input))
  return preds

def get_question_data(question_id):
  title, possible_answers, crowd_forecast, correct_answer, correct_forecast, open_date, close_date = get_question_page(question_id)
  preds = get_forecasts(question_id)
  question_id=question_id
  return {"question_id": question_id,
          "title": title,
          "possible_answers": possible_answers,
          "crowd_forecast": crowd_forecast,
          "correct_answer" : correct_answer,
          "correct_forecast": correct_forecast,
          "preds": preds,
          "open_date": open_date,
          "close_date": close_date}

def ids_file(path, last_page_no):
  page_no = 1
  while (page_no <= last_page_no):
    with open(path, 'a') as f:
      driver.get(f"https://www.gjopen.com/questions?status=resolved&type=forecasting&page={page_no}")
      soup = BeautifulSoup(driver.page_source, 'html.parser')  
      rows = soup.find_all(class_="row-table-row")
      for row in rows:
        link = row.find("h5").find("a").get("href")
        question_id = int(re.match(r"https:\/\/www\.gjopen\.com\/questions\/(\d+)", link).groups()[0])
        f.write(str(question_id))
        f.write('\n')
      page_no +=1
  f.close()

def get_all_questions(question_ids_file, path):
  with open(question_ids_file, "r") as f:
    for line in f.readlines():
      data = get_question_data(int(line))
      name = f"{path}question_{int(line)}.json"
      with open(name, "w") as outfile:  
        json.dump(data, outfile) 

# id_file = os.path.expanduser("~/your/path/to/id/file")
# last_page_no = 90
# save_path = os.path.expanduser("~/your/path/to/data/")

# ids_file(id_file, last_page_no)
# get_all_questions(id_file, save_path)