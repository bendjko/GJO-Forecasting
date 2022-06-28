from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json 
import os
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
driver.get("https://www.gjopen.com")
username = driver.find_element(By.XPATH, "//*[@id='user_email']")
password = driver.find_element(By.XPATH, "//*[@id='user_password']")

username.send_keys("ben.k@wustl.edu")
password.send_keys("qJ7C64Gb!CDqBsu")

driver.find_element(By.XPATH, "//*[@id='new_user']/input[2]").click()

def get_dates(question_id):
  link_template = f"https://www.gjopen.com/questions/{question_id}"
  driver.get(link_template)
  soup = BeautifulSoup(driver.page_source, 'html.parser')
  date = soup.find("div", class_="question-openclose").find_all("small")
  open_date = date[0].find("span").get("data-localizable-timestamp")
  close_date = date[1].find("span").get("data-localizable-timestamp")
  return open_date, close_date


def append_dates(question_ids_file, path):
  with open(question_ids_file, "r") as f:
    for line in f.readlines():
      open_date, close_date = get_dates(int(line))
      data_file = f"{path}question_{int(line)}.json"
      with open(data_file) as append_file:  
        jfile = json.load(append_file)
      # jfile.append({
      #     "open_date": open_date,
      #     "close_date": close_date
      # })
        jfile["open_date"] = open_date
        jfile["close_date"] = close_date
      with open(data_file, 'w') as json_file:
        json.dump(jfile, json_file)
        # print(jfile)


# id_file = os.path.expanduser("~/Desktop/id_file_modified.txt")
# save_path = os.path.expanduser("~/Desktop/data/")

# testing with question_2418.json

test_data = os.path.expanduser("~/Desktop/test_id.txt")
test_path = os.path.expanduser("~/Desktop/")

append_dates(test_data, test_path)

