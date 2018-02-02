import os
import numpy as np 
import sys
import urllib
import math
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf 
sc = SparkContext.getOrCreate()
import re
from collections import Counter
from string import punctuation
from nltk.stem import PorterStemmer

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
    
    x=re.sub('\.|,|:|;|\'|\!|\?|"','',x)
    #x=re.sub('!|"|#|$|%|&|\'|(|)|*|+|,|-|.|/|:|;|<|=|>|?|@|[|\|]|^|_|`|{|||}|~','',x)

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
        if(x[i] not in stopWords.value and (len(x[i])>2)):
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

######################################################################################################################
#def create_array(x):
#    ccat=
######################################################################################################################
def create_array1(x,a,b,c,d):
    
    ccat=1
    ecat=1
    gcat=1
    mcat=1
    
    for k in range(0,len(x)):
                        
        if((x[k])=='c'):
            ccat=ccat+(float)(x[k+1])
        elif((x[k])=='e'):
            ecat=ecat+(float)(x[k+1])
        elif((x[k])=='m'):
            mcat=mcat+(float)(x[k+1])
        elif((x[k])=='g'):
            gcat=gcat+(float)(x[k+1])


    #if(ccat>0):
    ccat=math.log(ccat/(a+vocab.value))
    #if(ecat>0):
    ecat=math.log(ecat/(c+vocab.value))
    #if(mcat>0):
    mcat=math.log(mcat/(b+vocab.value))
    #if(gcat>0):
    gcat=math.log(gcat/(d+vocab.value))
    '''
    if(ccat>0):
        ccat=math.log(ccat/a)
    if(ecat>0):
        ecat=math.log(ecat/c)
    if(mcat>0):
        mcat=math.log(mcat/b)
    if(gcat>0):
        gcat=math.log(gcat/d)
    '''
    return [ccat,mcat,ecat,gcat]
def predict_class(x,a,b,c,d):
    class_values=group_by_class.value    

    
    product=[0,0,0,0]
    for cc,v in x.items():
        word=cc
        
            
            
        ccat=0
        ecat=0
        gcat=0
        mcat=0
            
        
        for j in range(0,1):
            if(word in class_values):
                ccat=(float)(class_values[word][0])
                mcat=(float)(class_values[word][1])
                ecat=(float)(class_values[word][2])
                gcat=(float)(class_values[word][3])


               
            
        
        product[0]=product[0]+(ccat*((float)(v)))
        product[1]=product[1]+(mcat*((float)(v)))
        product[2]=product[2]+(ecat*((float)(v)))
        product[3]=product[3]+(gcat*((float)(v)))
    product[0]=product[0]+a
    product[1]=product[1]+b
    product[2]=product[2]+c
    product[3]=product[3]+d
    best=np.argmax(product)
    
    
    if(best==0):
        return "CCAT"
        #return str(product[0])
    elif(best==1):
        return "MCAT"
        #return str(product[1])
    elif(best==2):        
        return "ECAT"
        #return str(product[2])
    else:
        return "GCAT"
        #return str(product[3])


            
            

def read_stop_words():
    with open("stopWordList_2.txt") as f:
        readStopWords = [x.strip('\n').encode("utf-8") for x in f.readlines()]
        for i in range(0,len(readStopWords)):
            readStopWords[i]=str(readStopWords[i], "utf-8")

    return readStopWords
if __name__ == "__main__":


    
    if len(sys.argv) != 5:
        print("Usage: project_1.py <url_train> <url_test> <vsmall/small/large> <1>", file=sys.stderr)
        exit(-1)
    spark = SparkSession\
        .builder\
        .appName("naive_bayes")\
        .getOrCreate()
    
    #create stopwords later 
    stopWords = sc.broadcast(read_stop_words())
    #print(stopWords.value)
    base_url_train=(sys.argv[1])
    base_url_test=(sys.argv[2])
    dataset_size=(sys.argv[3])
    flag=((int)(sys.argv[4]))
    
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
    
    
    X_train=X_train.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    X_test=X_test.zipWithIndex().map(lambda x:(x[1],fix_doc(str(x[0], "utf-8"))))
    
    y_train=(y_train.map(getcats)\
        .zipWithIndex().map(lambda x:(x[1],x[0]))).collect()
    #splitting the X_train and y train's with multiple copies
    y_train=sc.parallelize(split_data(y_train))
    
    
    train_complete=X_train.join(y_train).map(lambda x:(x[1][0],x[1][1]))    
    
    print("check1")
    X_train=train_complete.map(lambda x:x[0])
    
    y_train=train_complete.map(lambda x:x[1])
   
    print("check1.25")
    if(flag==0):
       
        y_test=(sc.textFile("y_to_test.txt").map(lambda x:x.encode("utf-8")))
    
        x=(str("wget "+base_url_test+"y_test_"+dataset_size+" -O y_to_test.txt"))
        os.system(x)
        y_test=(y_test.map(getcats)\
            .zipWithIndex().map(lambda x:(x[1],x[0]))).collect()
        y_test=sc.parallelize(split_data(y_test))
        test_complete=X_test.join(y_test).map(lambda x:(x[1][0],x[1][1]))
        X_test=test_complete.map(lambda x:x[0])
        y_test=test_complete.map(lambda x:x[1])
       

    
    temp=y_train.map(lambda x:(x,1)).reduceByKey(lambda a,b:a+b).collect()
    print (temp)
    tot_ccat=temp[0][1]
    tot_mcat=temp[1][1]
    tot_ecat=temp[2][1]
    tot_gcat=temp[3][1]

    

    
    
    prior_ccat=(math.log(((float)(tot_ccat))/((float)(tot_ccat+tot_gcat+tot_mcat+tot_ecat))))
    prior_gcat=(math.log(((float)(tot_gcat))/((float)(tot_ccat+tot_gcat+tot_mcat+tot_ecat))))
    prior_ecat=(math.log(((float)(tot_ecat))/((float)(tot_ccat+tot_gcat+tot_mcat+tot_ecat))))
    prior_mcat=(math.log(((float)(tot_mcat))/((float)(tot_ccat+tot_gcat+tot_mcat+tot_ecat))))
    #del(temp)
    
    print("check2")
    vocab=(X_train.flatMap(lambda x:x.split(" ")).map(lambda x:re.sub(",","", x)).distinct().map(lambda x:(1,1)).reduceByKey(lambda a,b:a+b).collect())[0][1]
    #print(vocab)
    vocab=sc.broadcast(vocab)
    

   
    
    group_by_class=train_complete.map(lambda x:(x[1],x[0])).reduceByKey(lambda a,b:a+b).map(lambda x:([(x[0],z) for z in x[1].split(" ")]))\
                .flatMap(lambda x:(x)).map(lambda x:(x,1)).reduceByKey(lambda a,b:a+b)
    

    
    temp=group_by_class.map(lambda x:(x[0][0],1)).reduceByKey(lambda a,b:a+b).collect()
    print (temp)
    
    tot_ccat=temp[0][1]
    tot_mcat=temp[1][1]
    tot_ecat=temp[2][1]
    tot_gcat=temp[3][1]

    #build on this line
    group_by_class=group_by_class.map(lambda x:(x[0][1],(str(x[0][0])+" "+str(x[1])+" "))).reduceByKey(lambda a,b:str(a)+str(b))\
                .map(lambda x:(x[0],str(x[1]).replace("cat","").split(" "))).map(lambda x:(x[0],create_array1(x[1],tot_ccat,tot_mcat,tot_ecat,tot_gcat))).collect()

    #print((group_by_class.take(4)))

    group_by_class=sc.broadcast(dict(group_by_class))
    
    
    



    
    
    
    print("check3")
    
    #print(X_test.collect())
    

    y_test=X_test.map(lambda x:Counter(x[1].split(" "))).map(lambda x:predict_class(x,prior_ccat,prior_mcat,prior_ecat,prior_gcat))
    print("lastcheck")
    
    submit=open('y_test.txt',"w")
    y_test=y_test.collect()
    counter=0
    for i in y_test:
        counter=counter+1
        #print(counter)
        submit.write("%s\n" % i)
    submit.close()