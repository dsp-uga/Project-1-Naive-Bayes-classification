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
    #The output of this code would create a pair ((document,word),0) pair for each
    #possible combination of document and word by taking a cartisian.
    document_term_matrix = document_prior_lables.cartesian(bag_of_words_wonly)
    document_term_matrix = sc.parallelize(sorted(document_term_matrix.collect()))
    document_term_matrix = document_term_matrix.map(lambda word:(word,0))
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

"""
This function creates a model for the naive-bayse. 
@param document_term_matrix is the document term matrix got from broadcast variable.
@return naive bayes model which contains P(Wi/Cj), which is probability of each word i in document j.
"""
def naive_bayes(document_term_matrix):
    #Get the broadcast variable of document term matrix.
    document_term_matrix_rdd = sc.parallelize(document_term_matrix.value)
    
    #Subset the document term matrix to (doc lable,count) rdd.
    #Then calcuate number of words in each document lable.
    document_word_count = document_term_matrix_rdd.map(lambda x : (x[0][0],x[1]))
    document_word_count = document_word_count.reduceByKey(add)

    #Now find the posterior for each word by deviding a word count in a document by
    #total number of the words in the document.
    
    #Convert the pair ((doc lable, word), count) into (doc lable,(word,count)). 
    #It would help to implement a join.
    naive_bayes_model = document_term_matrix_rdd.map(lambda x:(x[0][0],(x[0][1],x[1])))
    #Join the DTM with the document word count. It would give the pair (doc lable, (word,count), document word count.)
    #e.g. ('ccat',('aaa',5),5000)
    naive_bayes_model = naive_bayes_model.join(document_word_count)
    #Devide the word count in the document with total number of words in the document. 
    #Also, conver the pair into ((doc lable,word), probability of a word i in doc j. P(Wi/Cj))
    #e.g. (('mcat','aaa'),00021258503401360543)
    naive_bayes_model = naive_bayes_model.map(lambda x:((x[0],x[1][0][0]),x[1][0][1]/x[1][1]))
    #Broadcast
    naive_bayes_model = sc.broadcast(naive_bayes_model.collect())
    return naive_bayes_model





"""
Given the document priors, test data and the naive bayes predictor data, this method predicts the output.
It uses the log probabilities to remove possibilities of underflow of numbers due to very small probability numbers.
@param test_data is the testing data set.
@param document_prior is the document lable prior probabilites got from broadcast variable.
@param naive_bayes_model is again a broadcast variable which gives the word probability given a document. 
@return predicted classes in (index,lable) pair.
"""
def predict_naive_bayes(test_data,document_prior,naive_bayes_model):
    #Get the doc,word count for each document and word.
    
    #Get the test text data and zip it with index.Make the index as key.
    predict_naive_bayes = test_data.zipWithIndex().map(lambda text: (text[1],text[0]))
    #Split the text by Space, which would give a pair (index, [list of words])
    predict_naive_bayes = predict_naive_bayes.map(lambda text:(text[0],text[1].split(" ")))
    #Separate each row in (index,word) format.
    predict_naive_bayes = predict_naive_bayes.flatMapValues(same_val)
    #Remove the row with the word size less than 2.
    predict_naive_bayes = predict_naive_bayes.filter(lambda text: len(text[1]) > 1)
    #Pad the pair by 0. Convert into a pair ((index,word),0)
    predict_naive_bayes = predict_naive_bayes.map(lambda word:(word,0))
    #Sum up the pairs with similar(index,word) key.
    predict_naive_bayes = sorted(predict_naive_bayes.countByKey().items())
    #Parallelize.
    predict_naive_bayes = sc.parallelize(predict_naive_bayes)
    
    #Get the document prior lables and create a cartisian of the them to have possible combinations
    #of (index,word) - lable.
    
    #Get the document prior from the 
    document_prior_lables = sc.parallelize(document_prior.value)
    #Get the lables only, ignore probabilities.
    document_prior_lables = document_prior_lables.map(lambda lable: (lable[0]))
    #Cartesian with the current list. Should give a pair like (((0, 'added'), 1), 'mcat'). (((index,word),count),lable).
    predict_naive_bayes = predict_naive_bayes.cartesian(document_prior_lables)
    
    
    #Convert the data pair into ((lable,word),(index,count),doc-word probability) by merging with 
    #naive bayes likelihood probabilities.
    #e.g. (('mcat', 'good'), ((0, 1), 0.0008503401360544217))
    
    #Update the structure into ((lable,word),(index,count))
    predict_naive_bayes = predict_naive_bayes.map(lambda x : ((x[1],x[0][0][1]),(x[0][0][0],x[0][1])))
    #Get naive bayes model.
    naive_bayes_model = sc.parallelize(naive_bayes_model.value)
    #Join the current list with the naive bayes model to convert the data into the desired format.
    predict_naive_bayes = predict_naive_bayes.join(naive_bayes_model)

    #Get the log-sum of all the probabilities.
    # Log (P(Cj)) + SUM ( Log (P(Wi/Cj)))
    
    #Get the summed log-probability of each word in the document for each lable.
    #e.g. ((0, 'ccat'), -372.50010540505934)
    
    #Multiply the log(P(Wi/Cj)) with the number of words.
    predict_naive_bayes = predict_naive_bayes.map(lambda x: ((x[1][0][0],x[0][0]),x[1][0][1] * math.log(x[1][1])))
    #Sum all the log-sum for (index,lable) key. e.g. ((0, 'mcat'), -399.2261060687056)
    predict_naive_bayes = predict_naive_bayes.reduceByKey(add)
    #Sort the lables.
    predict_naive_bayes = sorted(predict_naive_bayes.collect())
    #Parallelize.
    predict_naive_bayes = sc.parallelize(predict_naive_bayes)
    
    
    #Add the document prior probability log value. For that need to join the current list
    #with the document prior list.
    
    #Make the document lable as the key.
    predict_naive_bayes = predict_naive_bayes.map(lambda x : (x[0][1],(x[0][0],x[1])))
    #Get the document priors.
    document_prior_rdd =  sc.parallelize(document_prior.value)
    #Join with our list. The output should be (lable, ((index,log probabilities),document prior))
    # e.g. ('mcat', ((3, -741.0503284960512), 0.15584415584415584))
    predict_naive_bayes = predict_naive_bayes.join(document_prior_rdd)
    #Add the doc-prior log probability to the sum and sort. The pair should look like
    #e.g. (0, (-373.3781749241133, 'ccat'))
    predict_naive_bayes = predict_naive_bayes.map(lambda x : (x[1][0][0],(x[1][0][1] + math.log(x[1][1]),x[0])))
    predict_naive_bayes = sc.parallelize(sorted (predict_naive_bayes.collect()))
    
    #Now get the maximum log probabilities for the document.
    predict_naive_bayes = predict_naive_bayes.reduceByKey(max)
    #Keep the (index,lable) onlye. e.g. (0,'ccat') and sort the list.
    predict_naive_bayes = sorted(predict_naive_bayes.map(lambda x : (x[0], x[1][1])).collect())
    
    
    #Broadcast the results.
    predict_naive_bayes = sc.broadcast(predict_naive_bayes)
    return predict_naive_bayes

"""
Calculate the accuracy of the predicted results. => correct predicted lables/total lables.
@param test_lable is the list of the actual test lables.
@param predict_naive_bayes is the list of the predicted test lables.
@return the accuracy numbers.
"""
def result_analysis(test_lable,predict_naive_bayes):
    #Index the test lables.
    test_lable_rdd= test_lable.zipWithIndex().map(lambda x : (x[1],x[0]))
    #Get the join of the predicted and the actual lables.
    accuracy_list = sc.parallelize(predict_naive_bayes.value).join(test_lable_rdd)
    #Filter the list to have only lables matching with output.
    accuracy_list = accuracy_list.filter(lambda lable : lable[1][0] == lable[1][1])
    #Get the accuracy numbers.
    accuracy = accuracy_list.count()/test_lable_rdd.count()
    return accuracy



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
This method runs the whole naive-bayes method given the data set to fetch from the server and returns
the prediction result of the corpus.
@param data_set is either 'vsmall','small' or 'large'.
@return is the prediction result in (Zip Index of test document, 'Lable') format.
"""
def run_naive_bayes(data_set):
    #1. Fetching and cleaning data.
    #Create the Spark Config and Context.
    conf = SparkConf().setAppName('P1NaiveBayes')
    sc = SparkContext.getOrCreate(conf=conf)

    #Fetch the stopwords into a list and broadcast them across.
    stop_words = sc.broadcast(read_stop_words('stopWords.txt'))

    #Get the files and load them into memory.
    #To be done: Pass the URL-File Name via runtime arguments. 
    arg_url_train = 'https://storage.googleapis.com/uga-dsp/project1/train/'
    arg_text_train = 'X_train_' + data_set + '.txt'
    arg_lable_train = 'y_train_' + data_set + '.txt'
    train_text_log = 'training_data_log.txt'
    train_lable_log = 'training_lables_log.txt'
    
    arg_url_test = 'https://storage.googleapis.com/uga-dsp/project1/test/'
    arg_text_test = 'X_test_' + data_set + '.txt'
    arg_lable_test = 'y_test_' + data_set + '.txt'
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
    #Remove the lables except the ones having 'cat'
    train_lable = train_lable.map(remove_lables)
    #Apply Index to each row.
    train_lable = train_lable.zipWithIndex().map(lambda lable:(lable[1],lable[0])).collect()
    #Get it into an RDD.
    train_lable = sc.parallelize(train_lable)
    #Convert each lable row with multiple rows into separate rows with single lable.
    train_lable = train_lable.filter(lambda lable: len(lable[1])>0).flatMapValues(same_val)
    #Join the training data and the lables and create one to one training-lable set.
    train_join = train_data.join(train_lable).map(lambda join:(join[1][0],join[1][1]))
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
    
    #3. Create the Naive-Bayse Model.
    naive_bayes_model = naive_bayes(document_term_matrix)
    
    #4 Predict the classes.
    #Repeat the same process for the test as of training data.
    #Fetch the text-lable testing files and load.
    fetch_data(arg_url_test,arg_text_test,test_text_log)
    fetch_data(arg_url_test,arg_lable_test,test_lable_log)


    #Create RDDs for the fetched files. 
    #Also Convert files into the "UTF-8" format for convinience.
    #http://spark.apache.org/docs/2.1.0/api/python/pyspark.html
    test_data = sc.textFile(arg_text_test).map(lambda x:x.encode("utf-8"))
    test_lable= sc.textFile(arg_lable_test).map(lambda x:x.encode("utf-8"))


    #Clean the testing text file and index it with the zip with index method. 
    test_data = test_data.zipWithIndex().map(lambda text:(text[1], clean_data(str(text[0],'utf-8'))))

    #Remove the lables except the ones having 'cat'.
    test_lable = test_lable.map(remove_lables)
    #Apply Index to each row.
    test_lable = test_lable.zipWithIndex().map(lambda lable:(lable[1],lable[0])).collect()
    #Get it into an RDD.
    test_lable = sc.parallelize(test_lable)
    #Convert each lable row with multiple rows into separate rows with single lable.
    test_lable = test_lable.filter(lambda lable: len(lable[1])>0).flatMapValues(same_val)
    #Join the testing data and the lables and create one to one testing-lable set.
    test_join = test_data.join(test_lable).map(lambda join:(join[1][0],join[1][1]))
    #Get the testing and lables back again from the join.
    test_data = test_join.map(lambda join:join[0])
    test_lable = test_join.map(lambda join:join[1])
    #Predict the classes for the test data.
    predict_naive_bayes = predict_naive_bayes(test_data,document_prior,naive_bayes_model)
    
    #5.Calculate the accuracy.
    #accuracy = result_analysis(test_lable,predict_naive_bayes)

    
    return predict_naive_bayes

