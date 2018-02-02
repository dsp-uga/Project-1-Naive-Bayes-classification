# Project-1-Naive-Bayes-classification
DSP project 01 - Document classification using Naive Bayes:
Given the huge text data-sets this project tries to classify them into certain labels, namely (MCAT,CCAT,GCAT,ECAT). The data is from the Reuters' corpus, with having number of documents in about a million records.
The primary task of this assignment is to train a Naive-Bayes model for classifying the documents into different lables and check the accuracy on a new testing dataset from the similar domain. 
The classification is done in step-wise manner, which is as listed below:
1. Pre-processing: Remove all the punctuation marks and special characters. Also remove stopwords. The stopwords are take from the Stanford stop-words library. 
Reference: https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
2. Data Fixes: For one text document, there can be multiple lables in the corpus. So, it is needed to be resolved by creating a duplicate text document and assign each of them a single lable.
3. Creating Priors and Data-Dictionary: As the naive-bayes requires document prior (P(cj)) and the word priors (P(wi/cj) is the probability of the word i in the document j), the cleaned data is converted into an RDD of words and counts/probabilities for creating data-dictionary.  The document prior is calculated based on the number of lables of each set in the corpus.
4. Calculate Naive Bayes : It is fairly simple to calculate naive-bayes from given priors, but due to avoid underflow the probabilities are converted into log-probabilities. 
5. Accuracy: The predicted outputs are compared to the actual lables in terms of small and very small datasets to find accuracy of it. 




#CodeBase:

There are two branches of the codes:
1. master: Which contains codes created-updated by Maulik and Aishwarya.
How to run:  $SPARK_HOME/bin/spark-submit  <Project-Home>/gru/p1/run.py
Bugs : The large dataset runs crashes on small machines. The parameter for the data-set is hardcoded and should be converted to command-line argument.

2. test-1 : Which contains codes created-updated by Raunak and Aishwarya.
How to run:
 > python3 preprocessing.py <train_url_location> <test_url_location> <dataset_size> <0-you have y_test else 1 if you dont>
 Further Informamtion: The following code preprocess the text and lables set for the large/small/vsmall sets. 
 It stores it under - ``` 
 X_train_clean.txt, y_train_clean.txt, X_train_clean.txt and y_test_clean.txt (Optional) ```
 in the same folder where it is called. It will automatically collect the data and create these four files
 ***To delete the raw files please uncomment line number 189 and 190***
 If you have y_test i.e. the test lables please enter 0 in command line as the fourth parameter

 ***stopwords file is not added here, please use your custom stopwords since i am still building one***



#Contributors:
Maulik - Implementation of the Naive-Bayes and accuracy matrix, one separate Implementation of algorithm, documentation.
Raunak - Implementation of pre-processing and data-cleaning. Also, one separate implementation of the whole algorithm, documentation
Aishwarya - data-dictionary creation, Code Optimization, documentation.


#Future Works:
Implement N-grams inside this algorithm. 
Update the Stop-words for better performance. 
Better way to articulate the results. eg. F-Score. 

#References
http://spark.apache.org/docs/2.1.0/api/python/pyspark.html

https://stackoverflow.com/questions/3930188/how-to-convert-nonetype-to-int-or-string
https://stackoverflow.com/questions/32356017/generate-single-json-file-for-pyspark-rdd
https://github.com/dsp-uga/mauliknshah-p0

https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
https://stats.stackexchange.com/questions/163088/how-to-use-log-probabilities-for-gaussian-naive-bayes

