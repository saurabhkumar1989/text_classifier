# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:51:17 2016

@author: Ramanuja
"""

from pandas import read_csv
from collections import OrderedDict
import re,itertools
tweet_data = read_csv(filepath_or_buffer ="C:/Users/Ramanuja/Desktop/data2.csv",header=None,usecols = [9])# since no header info


tokenized_tweets = [re.findall(r'\w+',i) for i in tweet_data[9][1:]]# for tokenization

single_list = list(itertools.chain(*tokenized_tweets))# all list of token convert in to a single list

unique_word = list(OrderedDict.fromkeys(single_list))# remove duplicate

