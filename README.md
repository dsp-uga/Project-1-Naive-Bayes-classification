# Project-1-Naive-Bayes-classification
DSP project 01 - Document classification using Naive Bayes

### To run the preprocessing code type
 > python3 preprocessing.py <train_url_location> <test_url_location> <dataset_size> <0-you have y_test else 1 if you dont>

The following code preprocess the text and lables set for the large/small/vsmall sets. 
It stores it under X_train_clean.txt, y_train_clean.txt, X_train_clean.txt and y_test_clean.txt (Optional)
in the same folder where it is called. It will automatically collect the data and create these four files
To delete the raw files please uncomment line number 189 and 190
If you have y_test i.e. the test lables please enter 0 in command line as the fourth parameter