from itertools import islice
import numpy as np
import pandas as pd
import csv


def main():
    #Grab tweets from files and store them in a list
    #files are updated tweet_data_base_0_500.csv
    '''
    tweets = []
    for i in range(1, 500):
        tweets.append(open("updated_tweet_database_0_500.csv", "r", encoding="utf8").readlines()[i])
    
    for i in range(517, 991):
        tweets.append(open("updated_tweet_database517-991.csv", "r", encoding="utf8").readlines()[i])
    
    for i in range(1, 944):
        tweets.append(open("updated_tweet_database_0_944.csv", "r", encoding="utf8").readlines()[i])
    '''
    tweets = []
    with open("raw_data/updated_tweet_database_0_944.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        #only add rows 1 to 944 to tweets
        for row in islice(reader, 1, 944):
            #print(len(row))
            tweets.append(row)

    with open("raw_data/updated_tweet_database_0_500.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        #only add rows 1 to 500 to tweets
        for row in islice(reader, 1, 496):
            #print(len(row))
            tweets.append(row)

    with open("raw_data/updated_tweet_database517-991.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        #only add rows 517 to 991 to tweets
        for row in islice(reader, 517, 991):
            #print(len(row))
            tweets.append(row)
    with open("raw_data/updated_Tsla$TweetsNov.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        #only add rows 517 to 991 to tweets
        for row in islice(reader, 1, 324):
            #print(len(row))
            tweets.append(row)
    

    positive_tweets = 0
    negative_tweets = 0
    neutral_tweets = 0
    for n in tweets:
        if n[37] == "0.0":
            negative_tweets += 1
        elif n[37] == "1.0":
            neutral_tweets += 1
        elif n[37] == "2.0":
            positive_tweets += 1
        elif n[37] == 0.0:
            negative_tweets += 1
        elif n[37] == 1.0:
            neutral_tweets += 1
        elif n[37] == 2.0:
            positive_tweets += 1
        elif n[37] == "0":
            negative_tweets += 1
        elif n[37] == "1":
            neutral_tweets += 1
        elif n[37] == "2":
            positive_tweets += 1
        else:
            diego = 5 #do nothing

    print("Positive tweets: " + str(positive_tweets))
    print("Negative tweets: " + str(negative_tweets))
    print("Neutral tweets: " + str(neutral_tweets))

    #create a new column in tweets called relevance that adds up columns 16, 17, and 18
    for n in tweets:
        n.append(float(n[16]) + float(n[17]) + float(n[18]))

    tweets_filt = np.delete(tweets, [0,1,2,3, 6,7,8,9,10, 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36], axis=1)
    #count the amount of unique values in the first column and each unique value has a count for how many times it appears
    #print(np.unique(tweets[:,0], return_counts=True)) this prints the the amount of tweets per day
    print(tweets_filt[0])
    
    #randomize the rows of the tweets array
    np.random.shuffle(tweets_filt)
    #write the first 1000 rows of the randomized array to a test_data.csv file
    with open("dataset/tesla_stock_tweets/test_data.csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(tweets_filt[:1000])
    #write the next 1000 rows of the randomized array to a validate_data.csv file
    with open("dataset/tesla_stock_tweets/validate_data.csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(tweets_filt[1000:2000])
    #write the last rows of the randomized array to a train_data.csv file
    with open("dataset/tesla_stock_tweets/train_data.csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerows(tweets_filt[2000:])
    
    #write the neutral tweets to a neutral_tweets.csv file
    with open("database/test/neut/neutral_tweets.csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        for row in tweets_filt:
            if row[3] == "1.0" or row[3] == 1.0 or row[3] == "1":
                writer.writerow(row)
    #write the positive tweets to a positive_tweets.csv file
    with open("database/test/pos/positive_tweets.csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        for row in tweets_filt:
            if row[3] == "2.0" or row[3] == 2.0 or row[3] == "2":
                writer.writerow(row)
    #write the negative tweets to a negative_tweets.csv file
    with open("database/test/neg/negative_tweets.csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        for row in tweets_filt:
            if row[3] == "0.0" or row[3] == 0.0 or row[3] == "0":
                writer.writerow(row)





if __name__ == '__main__':
    main()