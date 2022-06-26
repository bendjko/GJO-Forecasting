from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
import json 
import time
import os

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
  r = requests.get(f"https://www.gjopen.com/questions/{question_id}")
  soup = BeautifulSoup(r.content, 'html.parser')
  date = soup.find("div", class_="question-openclose").find_all("span")
  open_date = date[0].get("data-localizable-timestamp")
  close_date = date[0].get("data-localizable-timestamp")
  return open_date, close_date

def get_all_dates(question_ids_file, path):
  with open(question_ids_file, "r") as f:
    for line in f.readlines():
      data = get_dates(int(line))
      name = f"{path}date_{int(line)}.json"
      with open(name, "w") as outfile:  
        json.dump(data, outfile) 

id_file = os.path.expanduser("~/Desktop/id_file_modified.txt")
save_path = os.path.expanduser("~/Desktop/data/")

get_all_dates(id_file, save_path)
