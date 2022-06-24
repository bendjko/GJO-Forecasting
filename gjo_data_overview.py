from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from selenium.webdriver.common.by import By
import requests
import re
import os
import json 
import time
from webdriver_manager.chrome import ChromeDriverManager



options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
# driver = webdriver.Chrome(executable_path='C:/path/to/chromedriver.exe')
driver.get("https://www.gjopen.com")

username = driver.find_element(By.XPATH, "//*[@id='user_email']")
password = driver.find_element(By.XPATH, "//*[@id='user_password']")
username.send_keys("ben.k@wustl.edu")
password.send_keys("qJ7C64Gb!CDqBsu")
driver.find_element(By.XPATH, "//*[@id='new_user']/input[2]").click()

def auto_scroll(question_id):
  link_template = f"https://www.gjopen.com/questions/{question_id}"
  driver.get(link_template)
  time.sleep(2) 

  SCROLL_PAUSE_TIME = 0.5

  # Get scroll height
  last_height = driver.execute_script("return document.body.scrollHeight")

  while True:
      # Scroll down to bottom
      driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

      # Wait to load page
      time.sleep(SCROLL_PAUSE_TIME)

      # Calculate new scroll height and compare with last scroll height
      new_height = driver.execute_script("return document.body.scrollHeight")
      if new_height == last_height:
          break
      last_height = new_height

  soup = BeautifulSoup(driver.page_source, "html.parser")

  return soup


def get_data(question_ids_file):
  forecast_count = 0
  written_justification_count = 0
  with open(question_ids_file, "r") as f:
    for line in f.readlines():
      question_id = line
      page = auto_scroll(question_id)
      for comment in page.find_all("div", class_="flyover-comment"):
        if "flyover-comment-load-more" not in comment.get("class") and "flyover-comment-reply" not in comment.get("class"):
            forecast_count += 1
            body = len(comment.find("div", class_="flyover-comment-content").find("p").text)
            if body != 0:
              written_justification_count +=1
  print(forecast_count, written_justification_count)

id_file = os.path.expanduser("~/Desktop/id_file.txt")
get_data(id_file)