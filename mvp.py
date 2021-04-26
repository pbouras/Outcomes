'''
from bs4 import BeautifulSoup
import requests
import pandas as pd
import html5lib
import pandas as pd
import requests

df = pd.read_html(requests.get('https://www.basketball-reference.com/leagues/NBA_2021_games-may.html').text, flavor="bs4")
df = pd.concat(df)
df.to_csv("new.csv", index=False)

'''

from bs4 import BeautifulSoup
import requests
import pandas as pd
import html5lib
import pandas as pd
import requests

games_data = []
years = ['2017','2018','2019','2020','2021']
count = 0

for year in years:

    months = ['december', 'january', 'february', 'march' , 'april', 'may']
    count = count + 1
    print(count)
    for month in months:
        df = pd.read_html(requests.get('https://www.basketball-reference.com/leagues/NBA_' + year + '_games-' + month + '.html').text, flavor="bs4")
        df = pd.concat(df)
        games_data.append(df)


#mydf = pd.DataFrame([games_data])
mydf = pd.concat(games_data, axis=0)
mydf.to_csv("new.csv", index = False)
#games_data.to_csv("new.csv", index=False)