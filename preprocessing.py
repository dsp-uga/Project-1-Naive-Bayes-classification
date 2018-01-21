import os
import numpy as np 
import urllib
from pyspark import SparkConf
from pyspark.context import SparkContext

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
import re
from collections import OrderedDict
#we are using python 3 soooo dont need to specify utf8 :>
#udf's 

#creates train_complete and test_complete which has the tuples in the directory format (X_train[0],y_train[0]) and (X_test[0],y_test[0])
#do not use this for use with large file since we dont know the y_test
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

def getcats(x):
    #scans and extract stuff with CAT/cat in suffix and makes it lower case
    x=str(x, "utf-8")
    x=x.split(",")
    fitered_data=[]
    for i in x:
        if ("CAT" or "cat") in i:
            fitered_data.append(i.lower())
    return fitered_data
        
    
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
        
def stop_words():
    new_list=[]
    with open("insert_name_here.txt") as f:
        readStopWords = [x.strip('\n').encode("utf-8") for x in f.readlines()]
    for i in range(0,len(readStopWords)):
        new_list.append(str(readStopWords[i],"utf-8"))
        
   

    return new_list

    
def main():

    sc = SparkContext(conf = SparkConf().setAppName("Project1_stuff"))
    if len(sys.argv) != 3:
        print("Usage: project_1.py <url_train> <url_test> <vsmall/small/large>", file=sys.stderr)
        exit(-1)
    
    #create stopwords later 
    #stopWords = sc.broadcast(read_stop_words())
    base_url_train=(sys.argv[1])
    base_url_test=(sys.argv[2])
    dataset_size=(sys.argv[3])
    
    #make local copy
    x=(str("wget "+base_url_train+"X_train_"+dataset_size+" -O X_to_train.txt"))
    os.system(x)

    x=(str("wget "+base_url_train+"y_train_"+dataset_size+" -O y_to_train.txt"))
    os.system(x)

    x=(str("wget "+base_url_test+"X_test_"+dataset_size+" -O X_to_test.txt"))
    os.system(x)

    x=(str("wget "+base_url_test+"y_test_"+dataset_size+" -O y_to_test.txt"))
    os.system(x)

    
    
    #we convert the text to utf-8 "the common unicode scheme" according to python docs
    X_train=(sc.textFile("X_to_train.txt").map(lambda x:x.encode("utf-8")))
    y_train=(sc.textFile("y_to_train.txt").map(lambda x:x.encode("utf-8")))
        
    X_test=(sc.textFile("X_to_test.txt").map(lambda x:x.encode("utf-8")))
    y_test=(sc.textFile("y_to_test.txt").map(lambda x:x.encode("utf-8")))
    #stopwords
    #stopWords = sc.broadcast(stop_words())
    #we are going to do what dr quinn mentioned and not use my crazy key gens
    #fix_doc is going to be the main cleanup function
    
    X_train=X_train.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    X_test=X_test.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    
    y_train=(y_train.map(getcats)\
        .zipWithIndex().map(lambda x:(x[1],x[0]))).collect()
    
    y_test=(y_test.map(getcats)\
        .zipWithIndex().map(lambda x:(x[1],x[0]))).collect()

    y_train=sc.parallelize(split_data(y_train))
    y_test=sc.parallelize(split_data(y_test))

    train_complete=X_train.join(y_train).map(lambda x:(x[1][0],x[1][1]))

    test_complete=X_test.join(y_test).map(lambda x:(x[1][0],x[1][1]))
    #splitting the X_train and y train's with multiple copies
    #splitting the X_test and if required uncomment the y_test
    X_train=train_complete.map(lambda x:x[0])
    y_train=train_complete.map(lambda x:x[1])
    X_test=test_complete.map(lambda x:x[0])
    #y_test=test_complete.map(lambda x:x[1])
    #create vocab
    vocab=X_train.flatMap(lambda x:x.split(" ")).map(lambda x:re.sub(",","", x)).distinct()


