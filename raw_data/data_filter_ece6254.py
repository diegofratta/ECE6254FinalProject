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
from pathlib import Path
import glob

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

ini_fin = [1000,1010] #Range of tweets to analyze
counter = 0
for i in np.arange(ini_fin[0], ini_fin[1]):
  if counter == 3:
    plt.close('all')
    clear_output(wait=True)
    counter = 0
  elif i%5 == 0: #Change the number to a number that works for you.
    aux = ini_fin[1] - ini_fin[0] - counter
    print(str(aux)+" tweets remains to finish.")
    uptd_db_path = ("./updated_tweet_database_"+
                    str(ini_fin[0])+"_"+str(ini_fin[1])+".csv") 
    #take note of the input range that you're doing if you change the name.
    files_present = glob.glob(uptd_db_path)
    # if no matching files, write to csv, if there are matching files, print statement
    if not files_present:
      tweets_db.to_csv(uptd_db_path)
      print('''
      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      \\\\\\\\\\\\---------------------------\\\\\\\\\\


      Saving file...


      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      ''')
    else:
      print('''
      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      \\\\\\\\\\\\---------------------------\\\\\\\\\\
      \\\\\\\\\\\\---------------------------\\\\\\\\\\


      WARNING: You're trying to rewrite the database!
      Choose another name.


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

0
tweets_db = pd.read_csv(filepath, dtype=str, sep=',').fillna(0) #This should point to the 
#new database with the labels... CHANGE
pos = 0
neut = 0
neg = 0
ph_t_n = "./database/train/neg/" #Path to train and negative
ph_t_p = "./database/train/pos/" #Path to train and positive
ph_t_nn = "./database/train/neut/" #Path to train and neutral
ph_e_n = "./database/test/neg/" #Path to test and negative
ph_e_p = "./database/test/pos/" #Path to test and positive
ph_e_nn = "./database/test/neut/" #Path to test and neutral

Path(ph_t_n).mkdir(parents=True, exist_ok=True)
Path(ph_t_p).mkdir(parents=True, exist_ok=True)
Path(ph_t_nn).mkdir(parents=True, exist_ok=True)
Path(ph_e_n).mkdir(parents=True, exist_ok=True)
Path(ph_e_p).mkdir(parents=True, exist_ok=True)
Path(ph_e_nn).mkdir(parents=True, exist_ok=True)

for i in range(ini_fin[0], ini_fin[1]+1):
  aux = "train" #We can change this afterwards, depending on the how do we want
  #to split the data... 
  tweet = tweets_db['tweet'].values[i]
  sv = tweets_db['sentimental_value'].values[i]

  if aux == "train":
    if sv == 0:
      with open(ph_t_n+str(neg)+"_"+str(i)+".txt", "w") as text_file:
        text_file.write(tweet)
      neg +=1
      pass
    elif sv == 1:
      with open(ph_t_nn+str(neut)+"_"+str(i)+".txt", "w") as text_file:
        text_file.write(tweet)    
      neut += 1
    elif sv == 2:
      with open(ph_t_p+str(pos)+"_"+str(i)+".txt", "w") as text_file:
        text_file.write(tweet)
      pos += 1
    else:
      print("sv = "+str(sv))
      print("tweet = "+str(tweet))
      print("i = "+str(i))
      print("Sentimental value out of range... skipping")

  elif aux == "test":
    if sv == 0:
      with open(ph_e_n+str(neg)+"_"+str(i)+".txt", "w") as text_file:
        text_file.write(tweet)
      neg +=1
      pass
    elif sv == 1:
      with open(ph_e_nn+str(neut)+"_"+str(i)+".txt", "w") as text_file:
        text_file.write(tweet)    
      neut += 1
    elif sv == 2:
      with open(ph_e_p+str(pos)+"_"+str(i)+".txt", "w") as text_file:
        text_file.write(tweet)
      pos += 1
    else:
      print("sv = "+str(sv))
      print("tweet = "+str(tweet))
      print("i = "+str(i))
      print("Sentimental value out of range... skipping")
  
  else: 
    print(str(aux)+" not a keyword, should be test or train instead")

!tar -czvf tweet_database.tar.gz ./database/
from google.colab import files
files.download("./tweet_database.tar.gz")
