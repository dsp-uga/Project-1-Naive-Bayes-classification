import os
import numpy as np 
import sys
import urllib
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf 
sc = SparkContext.getOrCreate()
import re
from collections import OrderedDict
#To run this code type python3 preprocessing.py <train_url_location> <test_url_location> <dataset_size> <0-you have y_test else 1 if you dont>
#The following code preprocess the text and lables set for the large/small/vsmall sets. 
#It stores it under X_train_clean.txt, y_train_clean.txt, X_train_clean.txt and y_test_clean.txt (Optional)
#in the same folder where it is called. It will automatically collect the data and create these four files
#To delete the raw files please uncomment line number 189 and 190
#If you have y_test i.e. the test lables please enter 0 in command line as the fourth parameter
########################################################################################################################
def fix_doc(x):
    #basic preprocessing - removing the same stuff from project 0 but later we will add more 
    #example we can import punctuations from string and use it but later :>
    #https://docs.python.org/2/library/re.html read more here
    
    x=re.sub('\.|,|:|;|\'|\!|\?','',x)
    #x=x.strip(".,:;'!?")
    
    # next we will do maulik's stuff before making it lower 
    # uncomment to use also maybe verify my logic 
    #x=re.sub("[ ([A-Z][a-zA-Z]*\s*)+]","",x) 
    
    #next we change to lower case 
    #and remove quote and amp
    x=x.lower()
    x=re.sub('&quot|&amp','',x)
    #last step remove everything which is not a letter and  
    x=re.sub('[^a-z]',' ',x)
    x=re.sub("\s\s+|\+"," ", x)
    a=""
    x=x.split(" ")
    for i in range(0,len(x)):
        if(x[i] not in stopWords.value):
            a=a+x[i]+" "
            
    a=a.strip()
    return a
#######################################################################################################################
def getcats(x):
    #scans and extract stuff with CAT/cat in suffix and makes it lower case
    x=str(x, "utf-8")
    x=x.split(",")
    fitered_data=[]
    for i in x:
        if ("CAT" or "cat") in i:
            fitered_data.append(i.lower())
    return fitered_data
        
######################################################################################################################  
def split_data(x):
    #splits the key value pair into multiple rows if the value has multiple values eg,
    #input [0,(first,second)] --> [[0,first],[0,second]]
    combined=[]
    
    for j in range(0,len(x)):
        
        key=x[j][0]
        value=x[j][1]
        
        for i in range(0,len(value)):
            single_row=[]
            single_row.append(key)
            single_row.append(value[i])
            combined.append(single_row)
    return (combined)
######################################################################################################################
def stop_words():
    #please use your own stopword file here
    new_list=[]
    with open("insert_name_here.txt") as f:
        readStopWords = [x.strip('\n').encode("utf-8") for x in f.readlines()]
    for i in range(0,len(readStopWords)):
        new_list.append(str(readStopWords[i],"utf-8"))
        
   

    return new_list

######################################################################################################################

            
            

if __name__ == "__main__":


    #sc = SparkContext(conf = SparkConf().setAppName("Project1_stuff"))
    if len(sys.argv) != 4:
        print("Usage: project_1.py <url_train> <url_test> <vsmall/small/large> <flag=0 if y_test is present 1 if not", file=sys.stderr)
        exit(-1)
    spark = SparkSession\
        .builder\
        .appName("pre_process")\
        .getOrCreate()
    
    
    #create stopwords later 
    #stopWords = sc.broadcast(read_stop_words())
    base_url_train=(sys.argv[1])
    base_url_test=(sys.argv[2])
    dataset_size=(sys.argv[3])
    flag=(sys.argv[4])
    
    #make local copy these are the raw files 
    x=(str("wget "+base_url_train+"X_train_"+dataset_size+" -O X_to_train.txt"))
    os.system(x)

    x=(str("wget "+base_url_train+"y_train_"+dataset_size+" -O y_to_train.txt"))
    os.system(x)

    x=(str("wget "+base_url_test+"X_test_"+dataset_size+" -O X_to_test.txt"))
    os.system(x)

 

    
    
    #we convert the text to utf-8 "the common unicode scheme" according to python docs
    X_train=(sc.textFile("X_to_train.txt").map(lambda x:x.encode("utf-8")))
    y_train=(sc.textFile("y_to_train.txt").map(lambda x:x.encode("utf-8")))
        
    X_test=(sc.textFile("X_to_test.txt").map(lambda x:x.encode("utf-8")))
    
    
    #fix_doc is going to be the main cleanup function for cleaning the text
    #getcats is the main function for cleaning the lables
    
    X_train=X_train.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    X_test=X_test.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    
    y_train=(y_train.map(getcats)\
        .zipWithIndex().map(lambda x:(x[1],x[0]))).collect()
    #splitting the X_train and y train's with multiple copies
    y_train=sc.parallelize(split_data(y_train))
    
    
    train_complete=X_train.join(y_train).map(lambda x:(x[1][0],x[1][1]))    
    
    
    X_train=train_complete.map(lambda x:x[0])
    y_train=train_complete.map(lambda x:x[1])
    #writing cleaned document to the X_train_clean.txt file
    submit=open('X_train_clean.txt',"w")
    X_train=X_train.collect()
    for i in X_train:
        submit.write("%s\n" % i)    
    submit.close()

    #writing lables into y_train_clean.txt files
    submit=open('y_train_clean.txt',"w")
    y_train=y_train.collect()
    for i in y_train:
        submit.write("%s\n" % i)    
    submit.close()

    if(flag==0):
        #we have both X_test and y_test -everything same as train sets
        y_test=(sc.textFile("y_to_test.txt").map(lambda x:x.encode("utf-8")))
    
        x=(str("wget "+base_url_test+"y_test_"+dataset_size+" -O y_to_test.txt"))
        os.system(x)
        y_test=(y_test.map(getcats)\
            .zipWithIndex().map(lambda x:(x[1],x[0]))).collect()
        y_test=sc.parallelize(split_data(y_test))
        test_complete=X_test.join(y_test).map(lambda x:(x[1][0],x[1][1]))
        X_test=test_complete.map(lambda x:x[0])
        y_test=test_complete.map(lambda x:x[1])
        submit=open('X_test_clean.txt',"w")
        X_test=X_test.collect()

        for i in X_test:
            submit.write("%s\n" % i)
    
        submit.close()
        submit=open('y_test_clean.txt',"w")
        y_test=y_test.collect()

        for i in y_test:
            submit.write("%s\n" % i)
    
        submit.close()
    else:
        #We just have the X_test set and we write it to file
        submit=open('X_test_clean.txt',"w")
        X_test=X_test.collect()

        for i in X_test:
            submit.write("%s\n" % i)
    
        submit.close()

#uncomment to delete the raw files
#del_raw=str("rm *_to_*.txt")
#os.system(del_raw)