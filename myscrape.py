from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import pandas as pd

driver = webdriver.Firefox(executable_path=r'C:\Users\panag\Scraper\drivers\geckodriver.exe')

years = ['2020-21','2019-20','2018-19','2017-18']
games_data = []

for year in years:

    driver.get('https://www.nba.com/stats/teams/traditional/?sort=W_PCT&dir=-1&Season='+ year +'&SeasonType=Regular%20Season')

    table = WebDriverWait(driver, 9).until(
        EC.presence_of_element_located((By.TAG_NAME, 'table'))
        )

    html = driver.page_source
    tables = pd.read_html(html)
    tables = pd.concat(tables)
    games_data.append(tables)

'''
driver.close()
#print(tables)
tables = pd.concat(tables)
tables.to_csv("mycsv.csv", index=False)
'''
mydf = pd.concat(games_data, axis=0)
mydf.to_csv("mycsv.csv", index = False)