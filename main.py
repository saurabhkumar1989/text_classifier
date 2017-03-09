# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:04:22 2017

@author: Ramanuja
"""
import sys,re
import logging
import itertools
import numpy as np
from random import random
from pandas import read_csv
from sklearn.svm import NuSVC
from collections import deque,Counter
import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def wordPlot(x,y,tweet,data_lables):
    #tweet parameter - to mark the words on graph
    labels = ['{0}'.format(i) for i in tweet]
    s = [500 for n in range(len(x))]
    plt.subplots_adjust(bottom = 0.1)
    text = ''
    for item in tweet:
        text = text + item + " "
    plt.xlabel(text)
    plt.scatter(
        x, y, marker='o', c=x, s=s,
        cmap=plt.get_cmap('Spectral'))
    
    for label, x, y in zip(labels, x, y):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Credit : Sklearn - http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py    
    """
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def process_tweets(tweet):
#    text =  text.decode('utf-8')
#    tknzr = TweetTokenizer()
    #Convert www.* or https?://* to space
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    
    #Convert @username to username
    tweet = re.sub('@([^\s]+)', r'\1',tweet)
    
    #Remove additional white spaces
    #tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub('#([^\s]+)', r'\1', tweet)
    
    #look for 2 or more repetitions of character and replace with the character itself
    #pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    #tweet = pattern.sub(r"\1\1", tweet)
    
    #Remove ... , -, special character
    #tweet = re.sub('[^A-Za-z]+', ' ', tweet)
    #trim
    tweet = tweet.strip('\'"')
    
    return re.split('\\W+',tweet)
def makeFeatureVec(words, model, num_features):
    
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    
    #Min case
    nwords = 1.0
    
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    

    data_points = []
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            data_points.append(model[word])# finding out the word vector reperesentation from a model
            #featureVec = np.add(featureVec,model[word])
    #clustering
    if len(data_points)>1:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data_points)
        class_labels = kmeans.labels_
        class_labels_count = Counter(class_labels)
        relevant_words = np.where(class_labels ==class_labels_count[max(class_labels_count.values())])# return the class having max point
        relevant_words = list(relevant_words[0])
        for j in relevant_words:
            featureVec = np.add(featureVec,data_points[j])
    else:
        for j in data_points:
            featureVec = np.add(featureVec,j)
    
        #data_.append(data_points[j])
    # Divide the result by the number of words to get the average
    #featureVec = np.divide(featureVec,nwords)
    return featureVec
def getAvgFeatureVecs(tweets, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")
    #
    # Loop through the reviews
    for tweet in tweets:
       
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(tweet, model, \
           num_features)
       
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs
def status(i,num_passe):
    barLength = 20 
    status = ""
    progress = (float(i)/(num_passe-1))
    block = int(round(barLength*progress))
    sys.stdout.write('\r')
    text = "[{0}] File Read {1}% Completed.".format( "#"*block + "-"*(barLength-block), format(progress*100,".2f"),status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def importTrainingData():
    # to train the classifier
    '''
    read data and tokenized it.
    data = [@anth0nypears0n Hi Anthony, no they don't. Children under 11 years travel 
     for free when travelling with a fare paying adult."]
    output = ['@anth0nypears0n', 'Hi', 'Anthony', no', 'they',.....],[]...]
    '''
    tweet_data = read_csv(filepath_or_buffer ="C:/Users/Ramanuja/Desktop/data_old.csv",header=None,skiprows=2,usecols = [1,2])# since no header info
    tweet_text = tweet_data[1]
    tweet_tag = tweet_data[2]
    filter_data = []
    filter_data_tag = []
    i = 0
    for text in tweet_text:
        #status(i,len(tweet_text))
        temp = process_tweets(text)
        if len(temp)>=1:
            filter_data.append(temp)
            filter_data_tag.append(tweet_tag[i])
        i = i + 1
    print "File Read Operation Completed"
    return filter_data,filter_data_tag
def importAnnotateedData():
    # to import the annotate data for word to vec training
    
    tweet_data = read_csv(filepath_or_buffer ="C:/Users/Ramanuja/Desktop/data_new.csv",header=None,skiprows=2,usecols = [1,2])# since no header info
    tweet_text = tweet_data[1]
    #tweet_tag = tweet_data[2]
    filter_data = deque()
    i = 1
    for text in tweet_text:
        #status(i,len(tweet_text))
        filter_data.append(process_tweets(text))
        i = i + 1
    print "File Read Operation Completed"
    return filter_data
    
# this is to train wor2vec model
X_train_word2vec = importAnnotateedData()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
class_names  = ["No Traffic info in Tweets","Traffic Info in Tweets"]
num_features = 3100    # Word vector dimensionality                      
min_word_count = 5   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-4   # Downsample setting for frequent words


model = word2vec.Word2Vec(X_train_word2vec, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
model_name = "firstModel"
model.save(model_name)
X, y = importTrainingData()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40, \
                                                    random_state=int(random()*100))


####### For Visulizing  ######################################################

#for visulization purpose
####### dimensional reduction usin isomap#####################################
#dmr = Isomap(n_neighbors=2, n_components=2, eigen_solver='auto', \
#             tol=0, max_iter=None, path_method='auto', \
#             neighbors_algorithm='auto', n_jobs=-1)
##gives al vocab in the model
#index2word_set = set(model.wv.index2word)
#no_of_tweets_for_visualization = 11
#tweets = X_test[:no_of_tweets_for_visualization+1]
#data_lables = []
#for tweet in tweets:
#    x = []
#    for word in tweet:
#        if word in index2word_set:
#            x.append(model[word])
#    kmeans = KMeans(n_clusters=2, random_state=0).fit(x) # as its a binary classifier so we want to divide the word in relevant and non relevant category
#    #below gives the class of lables
#    data_lables.append(kmeans.labels_)
#    wordPlot(data[:,0],data[:,1],tweet,data_lables)
## clustering different point
#
###### visulization end



##############################################################################


trainDataVecs = getAvgFeatureVecs( X_train, model, num_features )

testDataVecs = getAvgFeatureVecs( X_test, model, num_features )

## Neural Newtwork Classifier
clf = MLPClassifier(solver='sgd',shuffle=False, activation = 'logistic',\
                    alpha=1e-5,\
                     random_state=1,momentum =1e-4,max_iter=1000,tol =1e-10,\
                     verbose =True,early_stopping =False)

################################

### SVM classifier##############

clf = NuSVC(kernel="poly",tol =1e-10,verbose=True)

#################################

# Logistic Regression #########
clf = LogisticRegression(n_jobs =4,tol =1e-20,max_iter =10000,C=5.00)
################################

clf.fit(trainDataVecs, y_train)
y_pred = clf.predict(testDataVecs)
accuracy_score(y_test, y_pred,normalize=True)

accuracy = []
### K flod Validation#####################
number_of_folds = 10
for k in xrange(1,number_of_folds+1):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.40, random_state=int(random()*100))
    trainDataVecs = getAvgFeatureVecs( X_train, model, num_features )
    testDataVecs = getAvgFeatureVecs( X_test, model, num_features )
    clf.fit(trainDataVecs, y_train.get_values())
    y_pred = clf.predict(testDataVecs)
    accuracy.append(accuracy_score(y_test.get_values(), y_pred,normalize=True))
    print k
#########################################
#x_axis = np.arange(1.0, number_of_folds+1, 1)
#y_axis = accuracy
#plt.plot(x_axis, y_axis, 'go-')
#plt.axis([0, number_of_folds+1, 0.4, 1.0])
#plt.show()
#
## Compute confusion matrix
##cnf_matrix = confusion_matrix(y_test.get_values(), y_pred)
#
## Plot non-normalized confusion matrix
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names,
##                      title='Confusion matrix, without normalization')

