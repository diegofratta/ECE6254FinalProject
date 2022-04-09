# -*- coding: utf-8 -*-
"""data_filter_ECE6254.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EalOnEGDcHursKgRv0bKnCtO7EJUJ0A_
"""

import numpy as np
import pandas as pd
from PIL import Image
import requests
import re
import matplotlib.pyplot as plt
from IPython.display import clear_output

mm,dd,yyyy,close,volume,open,high,low = (np.loadtxt("./TeslaData5Years_mm_dd_yyyy.txt", 
                                                    unpack = True, skiprows=1, delimiter=","))

#
aux = close-open
aux[aux>0] = 1
aux[aux<=0] = 0
#More stuff to be added.

"""##Manual sentimental analysis of the tweets"""

filepath = "./tweet_data.txt"
tweets_db = pd.read_csv(filepath, dtype=str, sep=',').fillna(0)
df = pd.DataFrame()
df['sentimental_value'] = np.zeros(len(tweets_db['id'].values))
tweets_db = pd.concat([tweets_db, df], axis=1)

def show_tweet(tweet, photos):
  """
  inputs: 
  photos = output of get_urls
  tweet = tweet_db['tweet'].values[i]
  """
  print("Tweet:")
  print(tweet)
  for link in photos:
    if (len(link)>4):
      response = requests.get(link, stream=True)
      img = Image.open(response.raw)
      plt.imshow(img)
      plt.show()
    else:
      pass
  return None

def sentimental_evaluation():
  '''
  0 = bad
  1 = neutral
  2 = good
  '''
  print("Insert the sentimental value for the above tweet")
  print("0 = bad, 1 = neutral, 2 = good")
  while True:
    try:
      sv = int(input())
      if sv > 2 or sv < 0:
        raise ValueError
    except ValueError:
      print("Input is not 0, 1, 2")
      continue
    break
  print("***********--------------------------------***********")
  return sv

def get_urls(url):
  """Given the urls as a list of strings, will return an array with 
  the urls and some other characters that we will filter later
  input:
  url: tweets_db['photos'].values[i]
  """
  inputstring = url
  inputstring.replace('[', '')
  inputstring.replace(']', '')
  url_format = re.findall("([^']*)", inputstring)

  return url_format

ini_fin = [0,100] #Range of tweets to analyze
counter = 0
for i in np.arange(ini_fin[0], ini_fin[1]):
  if counter == 3:
    plt.close('all')
    clear_output(wait=True)
    counter = 0
  elif i%5 == 0: #Change the number to a number that works for you.
    aux = ini_fin[1] - ini_fin[0] - i
    print(str(aux)+" tweets remains to finish.")
    tweets_db.to_csv(r"./updated_tweet_database.csv")
    print('''
    \\\\\\\\\\\\---------------------------\\\\\\\\\\
    \\\\\\\\\\\\---------------------------\\\\\\\\\\
    \\\\\\\\\\\\---------------------------\\\\\\\\\\


    Saving file...


    \\\\\\\\\\\\---------------------------\\\\\\\\\\
    \\\\\\\\\\\\---------------------------\\\\\\\\\\
    \\\\\\\\\\\\---------------------------\\\\\\\\\\
    ''')
    df.to_csv(index=False)
  urls = get_urls(tweets_db['photos'].values[i])
  show_tweet(tweets_db['tweet'].values[i], urls)
  sv = sentimental_evaluation()
  tweets_db['sentimental_value'].values[i] = sv
  counter += 1

