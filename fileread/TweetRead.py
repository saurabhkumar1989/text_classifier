#from preProcessing.tweet_cleanser import process_tweets

#csv file read
from pandas import read_csv

import sys

def status(i,num_passe):
    barLength = 20 
    status = ""
    progress = (float(i)/(num_passe-1))
    block = int(round(barLength*progress))
    sys.stdout.write('\r')
    text = "[{0}] File Read {1}% Completed.".format( "#"*block + "-"*(barLength-block), format(progress*100,".2f"),status)
    sys.stdout.write(text)
    sys.stdout.flush()

def importData():
    tweet_data = read_csv(filepath_or_buffer ="C:/Users/Ramanuja/Desktop/data2.csv",header=None,skiprows=2,usecols = [9,10])# since no header info
    filter_data = []
    i = 0
    for text in tweet_data[9]:
        status(i,len(tweet_data[9]))
        i = i + 1
        #filter_data.append(process_tweets(text))
    y = tweet_data[10]
    #vector = bagofWords(filter_data,1)
    print "File Read Operation Completed"
    return filter_data,y
if __name__ == "__main__":
     data = ["this is for test","John likes to watch movies. Mary likes movies too.","John also likes to watch football games."]
     X,y =  importData()
