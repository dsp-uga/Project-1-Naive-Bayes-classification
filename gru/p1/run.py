# -*- coding: utf-8 -*-

"""
Created on Tue Jan 23 23:25:59 2018

@author: Maulik
"""

"""
This file is a consolidated file for the text analysis of the Reuter's text data. 
It contains the following steps to perform machine learning operation on the
text files.
1. Clean the text : Remove Stop Words,punctuations,trim the data. 
2. Create a Document Term Matrix: A matrix of word counts, where each word is
a row lable and the document is the column lable. 
3. Create a Naive Bayse model for the give training sets. 
4. Predict the output lables on the test sets given the Naive Bayse training 
model. 
5. As the test result is already available, compare the test results with the
predicted results and create a confusion matrix.
"""

#import libraries
from pyspark import SparkContext, SparkConf
import os
import numpy as np 
import urllib


#This function reads the Stop Words from the file containing stopwords. "stopWords.txt" 
#It uses a subset of the words given in the Stanford NLP library.
#@Ref https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
def read_stop_words():
    #Create a new list for the stopwords.
    stop_words_list=[]
    #Read all the words into an array.
    with open("stopWords.txt") as stop_word_file:
        read_stop_word = [stop_word.strip('\n').encode("utf-8") for stop_word in stop_word_file.readlines()]
    #Create a string list of all the stopwords. 
    for word in range(0,len(read_stop_word)):
        stop_words_list.append(str(read_stop_word[word],"utf-8"))
    #Return the Stopword list.    
    return stop_words_list





#Create the Spark Config and Context.
conf = SparkConf().setAppName('P1NaiveBayes')
sc = SparkContext.getOrCreate(conf=conf)

#Fetch the stopwords into a list and broadcast them across.
stop_words = sc.broadcast(read_stop_words())


