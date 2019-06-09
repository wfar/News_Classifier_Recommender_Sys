#!/usr/bin/env python

''' DataProcessor.py contains the singleton class to handle all data
    used for training, testing the model as well as user input data
    goes through here for vectorization and processing to be used and
    stored
    
'''

#builtin mods
import os
import sys
import time
import csv
import json
from sqlalchemy import create_engine
import sqlite3

#third party mods
import numpy as np
import pandas as pd
from sklearn import *

# put nay changes to path here!!
PATH_TO_HP = os.getcwd() + "\\huffpo\\News_Category_Dataset_v2.json" 


__author_ = "{wfar}"
__version__ = "{v1.0}"


class DataProcessor:


    def __init__(self):
        
        self.le = LabelEncoder()
        self.tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
        

    def __preprocess(self):

        data = data = pd.read_json(PATH_TO_HP, lines=True)                                           # read in the huffpost data
        data["text"] = data['headline'].map(str) + " " + data['short_description'].map(str)          # add a column text that combines headline and description columns


        X = data["text"]                                  # set X to hold the new txt column
        y = data['category'].apply(str)                   # set Y to hold the class label for the data category column)


        min_recs = min(data.groupby('category')['text'].nunique())          # find minimum number of records per class label
        balanced_data = data.groupby('category', as_index=False, group_keys=False).apply(lambda s: s.sample(min_recs + 10000,replace=True))   # randomly sameple (under and over) each class label so data is more balanced
        new_len_of_proc_data = len(self.balanced_data)       # save new number of records in data 
        
        X = balanced_data['text']                 # update and store the new X value
        y = balanced_data['category']             # update and store the new y value

        return balanced_data, X, y
    
    def __splitData(self, X, y):

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size =0.2, random_state=0)  # split data randomly into train and test split of 80/20% resp.

        return X_train, X_test, y_train, y_test
        
    def __vectorize(self, X, X_train, X_test):

        self.tfidf_vect.fit(X)
        X_train_tfidf =  tfidf_vect.transform(X_train)
        X_test_tfidf =  tfidf_vect.transform(X_test)

        return X_train_tfidf, X_test_tfidf
    
    def __encodeLabels(self):

        y_train_enc = self.le.fit_transform(y_train)
        y_test_enc = self.le.fit_transform(y_test)

        return y_train_enc, y_test_enc

    def input_vectorize(self):

        pass

    def __createDB(self):

        pass

    def get_classifier_train_info(self):

        try:
            balanced_data, X, y = self.__preprocess()
            X_train, X_test, y_train, y_test = self.__splitData(X, y)
            y_train_enc, y_test_enc = self.__encodeLabels(y_train, y_test)
            X_train_tfidf, X_test_tfidf = self.vectorize(X_train, X_test)

            return [self.le, X_train_tfidf, y_train_enc, X_test_tfidf, y_test_enc]
            
        except exception as e:
            print("Error: ", e)
        
        

        
