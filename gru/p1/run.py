# -*- coding: utf-8 -*-

"""
Created on Tue Jan 23 23:25:59 2018

@author: Maulik, Raunak, Aishwarya
"""

"""
This file is a consolidated file for the text analysis of the Reuter's text data. 
It contains the following steps to perform machine learning operation on the
text files.
1. Clean the text : Remove Stop Words,punctuations,trim the data.Keep the text with 'cat' lables only.
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
import re
from operator import add

"""
This function reads the Stop Words from the file containing stopwords. "stopWords.txt" 
It uses a subset of the words given in the Stanford NLP library.
@param file_name is the stop word filename with the correct path. 
@return List of stop words read from the file.
@ref https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
"""
def read_stop_words(file_name):
    #Create a new list for the stopwords.
    stop_words_list=[]
    #Read all the words into an array.
    with open(file_name) as stop_word_file:
        read_stop_word = [stop_word.strip('\n').encode("utf-8") for stop_word in stop_word_file.readlines()]
    #Create a string list of all the stopwords. 
    for word in range(0,len(read_stop_word)):
        stop_words_list.append(str(read_stop_word[word],"utf-8"))
    #Return the Stopword list.    
    return stop_words_list

"""
This function fetches the text files from the online URLs and loads it locally to
use further.
The URL and the file name is supposed to be passed to the function as parameters.
@param url is the online URL of the file location.
@param file_name is the text file name.
@param log_file is the wget logs to be saved in the file.
@return none.
"""
def fetch_data(url,file_name,log_file):
    #Create a string to get the file with wget command.
    wget_cmd = (str('wget ' + url + file_name + ' -o ' + log_file))
    #Run on the shell.
    os.system(wget_cmd)
    return

"""
This function cleans the text file with punctuations,special characters and extra white spaces.
Also, all the text is converted into lower cases and then the stop words are being removed.  
The text contains '&quot;' and '&amp;' special characters, which are also being removed. 
@param text_data is the original text data file in the string format.
@return filtered_text_data which is the required text output.

"""
def clean_data(text_data):
    #Remove special characters.
    text_data = re.sub('&quot;|&amp;',' ',text_data)
    
    text_data = text_data.lower()
    #Remove digits.
    text_data = re.sub('[0-9]+','',text_data)
    #text_data = re.sub(' [^a-z] ','',text_data)
    #Remove punctuation Marks
    text_data= re.sub(r'[()\.\,\:\;\'\"\!\?\-\+\\\$\/]',' ',text_data)
    #Remove Stop Words and Strip.
    #Filtered variable to return.
    filtered_text_data = ''
    #Split the data by space.
    text_data = text_data.split(' ')
    
    #Inefficient: To be updated.
    #For each string in the word, strip it and join again.
    for word in range(0,len(text_data)):
        if(text_data[word] not in stop_words.value):
            filtered_text_data = filtered_text_data + text_data[word].strip() + ' '
    
    #Strip the extra spaces.
    filtered_text_data = filtered_text_data.strip()
    return filtered_text_data

	
"""
This function removes the Y-Lables other than the lables having 'CAT' as their prefix or postfix.
@param lables is the set of the lables from the file in RDD element.
@return filtered lables in a list.
"""
def remove_lables(lables):
    #Convert the lables into utf-8 string.
    lables = str(lables,'utf-8')
    #Convert the lables into lowercase.
    lables = lables.lower()
    #Split the labes by ,
    lables = lables.split(',')
    #Filtered lables.
    filtered_lables = []
    #Keep the lables,which only contains 'cat'.
    #Inefficient code: Need to replace.
    for lable in lables:
        if 'cat' in lable:
            filtered_lables.append(lable)
    return filtered_lables


"""
Returns the same value as argument. Used for the flatMapValues function.
@param any value.
@return return the same value as in the argument.
"""
def same_val(val):
     return val	



"""
This function calculates the Bag of the words, which is the list of distinct words.
Here the bag of words broadcast variable also contains the overall count in the corpus.
@param train_data is the training data for which the bag of words needed to be calculated.
@return the bag of the words.
"""
def calculate_bag_of_words(train_data):
    #Split the data by spaces.
    bag_of_words = train_data.flatMap(lambda text: text.split(" "))
    #Strip the extra spaces and remove the words less than 2 characters long.
    bag_of_words = bag_of_words.map(lambda word: word.strip()).filter(lambda word: len(word)>1)
    #Create a (Word,Count) tuple and collect.
    bag_of_words = bag_of_words.map(lambda word: (word,1)).reduceByKey(add).collect()
    #Sort the vocabulary.
    bag_of_words = sorted(bag_of_words)
    #Broadcast this variable.
    bag_of_words = sc.broadcast(bag_of_words)
    return bag_of_words


"""
This function calculates the document prior,which means the proportion of each type of document
in the corpus. At the end broadcast the values.
@param train_lable is the training lables of the training corpus.
@return the document prior.
"""

def calculate_document_prior(train_lable):
    #Caculate the total count of each type of document in (lable,count) pair.
    document_prior = train_lable.map(lambda lable:(lable,1)).reduceByKey(add)
    #Find total number of documents.
    total_lables = train_lable.count()
    #Devide the count of each document with the total number of documents. Get Prior.
    document_prior = document_prior.map(lambda lable:(lable[0], lable[1]/total_lables)).collect()
    #Broadcast the document prior.
    document_prior = sc.broadcast(document_prior)
    return document_prior

"""
This function step by step calculates document term matrix. It goes by this:
1. Calculate the possible combination of words with the document labels. 
2. Calculate actual word-document combination.
3. Merge them up to create the matrix.
4. Broadcast this matrix.
@param train_join is the text-lable RDD for the training set.
@param document_prior is the broadcast variable of the document prior list.
@param bag_of_words is the broadcast variable for the bag of the words.
@return Document term matrix.
"""
def calculate_documente_term_matrix(train_join,document_prior,bag_of_words):
    #Get the Bag of the words. Words only RDD from the Broadcast variable.
    bag_of_words_wonly = sc.parallelize(bag_of_words.value)
    bag_of_words_wonly = bag_of_words_wonly.map(lambda lable: (lable[0]))
    #Get the lables from the Lable priors broadcast variable.
    document_prior_lables = sc.parallelize(document_prior.value)
    document_prior_lables = document_prior_lables.map(lambda lable: (lable[0]))
    #Create a cartisian of the words and the lable and add 1 count to each pair.
    #The output of this code would create a pair ((document,word),1) pair for each
    #possible combination of document and word by taking a cartisian.
    document_term_matrix = document_prior_lables.cartesian(bag_of_words_wonly)
    document_term_matrix = sc.parallelize(sorted(document_term_matrix.collect()))
    document_term_matrix = document_term_matrix.map(lambda word:(word,1))
    #Below code creates a word-document matrix with existance of such pair in the
    #document. Eg. (('ccat','aaa'),5)
    #The only words which are paired with the document are actually in that document.
    
    #Create('document lable','text') pair.
    document_word_count = train_join.map(lambda x: (x[1],x[0]))
    #Split the text into words.
    document_word_count = document_word_count.map(lambda x:(x[0],x[1].split(" ")))
    #Convert the (lable,list of words) into a separate (lable,word) pair for each word.
    document_word_count = document_word_count.flatMapValues(same_val)
    #Remove the rows, which have word length less than 2.
    document_word_count = document_word_count.filter(lambda word : len(word[1]) > 1)
    #Create a pair ((lable,word),0).
    document_word_count = document_word_count.map(lambda word:(word,0))
    #Calculate each (lable,word) pairs and sum them up. The pair would look like ((lable,word),counter)
    #e.g. (('ccat','aaa'),5)
    document_word_count = sorted(document_word_count.countByKey().items())
    document_word_count = sc.parallelize(document_word_count)


    #Merge the possible pairs with the actual pairs and create a document term matrix.
    
    #This would create a merged pair of ((lable,word),(1,actual count value or None))
    document_term_matrix = document_term_matrix.leftOuterJoin(document_word_count)
    #Sum up 1+ (Actual Counts). Replace None with 0. Now, the pair is ((lable,word),count)
    #The document term matrix is ready !!
    document_term_matrix = document_term_matrix.map(lambda x: (x[0],x[1][0] + int(0 if x[1][1] is None else x[1][1])))
    #Sort the matrix.
    document_term_matrix = sorted(document_term_matrix.collect())
    #Broadcast this matrix.
    #Need to check if the matrix is too large for the broadcast variable.
    document_term_matrix = sc.broadcast(document_term_matrix)
    return document_term_matrix




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



#1. Fetching and cleaning data.
#Create the Spark Config and Context.
conf = SparkConf().setAppName('P1NaiveBayes')
sc = SparkContext.getOrCreate(conf=conf)

#Fetch the stopwords into a list and broadcast them across.
stop_words = sc.broadcast(read_stop_words('stopWords.txt'))

#Get the files and load them into memory.
#To be done: Pass the URL-File Name via runtime arguments. 
arg_url_train = 'https://storage.googleapis.com/uga-dsp/project1/train/'
arg_text_train = 'X_train_vsmall.txt'
arg_lable_train = 'y_train_vsmall.txt'
train_text_log = 'training_data_log.txt'
train_lable_log = 'training_lables_log.txt'

arg_url_test = 'https://storage.googleapis.com/uga-dsp/project1/test/'
arg_text_test = 'X_test_vsmall.txt'
arg_lable_test = 'y_test_vsmall.txt'
test_text_log = 'test_data_log.txt'
test_lable_log = 'test_lables_log.txt'

#Fetch the text-lable training files and load.
fetch_data(arg_url_train,arg_text_train,train_text_log)
fetch_data(arg_url_train,arg_lable_train,train_lable_log)


#Create RDDs for the fetched files. 
#Also Convert files into the "UTF-8" format for convinience.
#http://spark.apache.org/docs/2.1.0/api/python/pyspark.html
train_data = sc.textFile(arg_text_train).map(lambda x:x.encode("utf-8"))
train_lable= sc.textFile(arg_lable_train).map(lambda x:x.encode("utf-8"))


#Clean the training text file and index it with the zip with index method. 
train_data = train_data.zipWithIndex().map(lambda text:(text[1], clean_data(str(text[0],'utf-8'))))

#Remove the lables except the ones having 'cat'.
train_lable = train_lable.map(remove_lables)
#Apply Index to each row.
train_lable = train_lable.zipWithIndex().map(lambda lable:(lable[1],lable[0])).collect()
#Get it into an RDD.
train_lable = sc.parallelize(train_lable)
#Convert each lable row with multiple rows into separate rows with single lable.
train_lable_f = train_lable.filter(lambda lable: len(lable[1])>0).flatMapValues(same_val)
#Join the training data and the lables and create one to one training-lable set.
train_join = train_data.join(train_lable).map(lambda join:(join[1][0],join[1][1])).flatMapValues(same_val)
#Get the training and lables back again from the join.
train_data = train_join.map(lambda join:join[0])
train_lable = train_join.map(lambda join:join[1])


#2. Document Term Matrix.
#Calculate Bag of Words.
bag_of_words = calculate_bag_of_words(train_data)
#Calculate Document Prior.
document_prior = calculate_document_prior(train_lable)
#Calculate the Document Term Matrix.
document_term_matrix = calculate_documente_term_matrix(train_join,document_prior,bag_of_words)
