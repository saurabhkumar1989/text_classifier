# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:51:17 2016

@author: Ramanuja
"""
from collections import OrderedDict,Counter
import re,itertools
import sys

def status(i,num_passe):
    barLength = 20 
    status = ""
    progress = (float(i)/(num_passe-1))
    block = int(round(barLength*progress))
    sys.stdout.write('\r')
    text = "[{0}] Text Vectorization {1}% Completed.".format( "#"*block + "-"*(barLength-block), format(progress*100,".2f"),status)
    sys.stdout.write(text)
    sys.stdout.flush()
def bagofWords(text_data,n=1):
    """ text data should be in array form 
    i.e. in arrya of list
    Ex :
    data = ["John likes to watch movies. Mary likes movies too.","John also likes to watch football games."]
    
    number of ngrams is decided by n
    by defalutl it is 1
    """
    unique_ngram = []
    
    single_list = []# sinlge list of all the Ngram - might have duplicate 
    
    tokenized_data = [re.findall(r'\w+',i) for i in text_data]# for tokenization
    
    
    single_list = list(itertools.chain(*tokenized_data))# all list of token convert in to a single list

    def listToNgram(single_list):
        ngram = []
        # if input is ['John', 'likes', 'to','watch']
        # output is ['John likes','likes to','to watch'] if n = 2
        if n>1:
            a = 1 #helpful when n = 2 as when n = 2 in ngram tokenization, last word be unigram so to avoid iterate only to len-1    
        else:
            a = 0
        for i in range(len(single_list)-a):
            words = single_list[i:i+n]# select words from a lsit according to ngram
            word = " ".join(words) #combine the words in to a single sentence
            ngram.append(word)
        return ngram
        
    ngrams = listToNgram(single_list)# contail all ngram from a input
    #print ngrams
    unique_ngram = list(OrderedDict.fromkeys(ngrams))# remove duplicate from ngram
    temp = []
    bow = []
    num_of_passes = len(tokenized_data)
    # convert input text array in to a n gram and then apply bag of word technique
    for i in range(num_of_passes):
        datax = listToNgram(tokenized_data[i])
        word_freq = Counter(datax) # have all the word - how many times it appearin the sentence
        temp = []
        for j in unique_ngram:
            if j in (datax):
                freq = word_freq[j]
                temp.append(freq)
            else:
                temp.append(0)
        bow.append(temp)
        status(i,num_of_passes)
    print "Text Vectorization Operation Completed"
    return bow
        
if __name__ == "__main__":
     data = ["this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","this is for test","John likes to watch movies. Mary likes movies too.","John also likes to watch football games."]
     bagofWords(data,1)
