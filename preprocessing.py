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
def to_unicode(x):
    #accept a set of input, converts to unicode the parts it can and ignores the rest.
    if type(x) != unicode:
        x =  x.decode('utf-8','ignore')
        #strip tuning for later as needed - i.e. fill in the () with ("stuff")
        return x.strip()
    else:
        
        return x.strip()
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
    return x

def getcats(x):
    
    x=x.split(",")
    fitered_data=[]
    for i in x:
        if ("CAT" or "cat") in i:
            filtered_data.append(i)
    return filtered_data
    
    

    
def main():

    sc = SparkContext(conf = SparkConf().setAppName("Project1_stuff"))
    if len(sys.argv) != 3:
        print("Usage: project_1.py <url_train> <url_test> <vsmall/small/large>", file=sys.stderr)
        exit(-1)
    
    #create stopwords later 
    #common_stop_words=sc.broadcast(sc.textFile("dothingsherelater.txt").collect())
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
    
    #we are going to do what dr quinn mentioned and not use my crazy key gens
    #fix_doc is going to be the main cleanup function
    
    X_train=X_train.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    X_test=X_test.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    
    #mapping X_train to y_train and X_test to y_test
    
    
    #y_train=ytrain.map(lambda x:x.split("\r\n")).map(getcats)\
    #        .lower().zipWithIndex().map(lambda x:x[1],x[0]).map(lambda x:OrderedDict((a,b) for (a,b) in x))
        
    
    

