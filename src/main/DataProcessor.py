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
import sqlite3 as db

#third party mods
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# put nay changes to path here!!
PATH_TO_HP = os.getcwd() + "\\huffpo\\News_Category_Dataset_v2.json" 


__author_ = "{wfar}"
__version__ = "{v1.0}"


class DataProcessor:

    #constructor
    def __init__(self):
        
        self.encoder = LabelEncoder()  # label encoder object used throughout run time
        self.vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)   # vecotizer object used throughout runtime

        self.model_info = self.__setupClf()    # sets up all the data and create list of trining info for models to use
        
    #private method
    def __preprocess(self):
        print("Start pp")
        data = data = pd.read_json(PATH_TO_HP, lines=True)                                           # read in the huffpost data
        data["text"] = data['headline'].map(str) + " " + data['short_description'].map(str)          # add a column text that combines headline and description columns
        print("data read")

        X = data["text"]                                  # set X to hold the new txt column
        y = data['category'].apply(str)                   # set Y to hold the class label for the data category column)

        print("x y created")
        min_recs = min(data.groupby('category')['text'].nunique())          # find minimum number of records per class label
        balanced_data = data.groupby('category', as_index=False, group_keys=False).apply(lambda s: s.sample(min_recs + 10000,replace=True))   # randomly sameple (under and over) each class label so data is more balanced
        new_len_of_proc_data = len(balanced_data)       # save new number of records in data 

        print("done resampling")
        
        X = balanced_data['text']                 # update and store the new X value
        y = balanced_data['category']             # update and store the new y value
        print("done pp")
        return balanced_data, X, y
    
    def __splitData(self, X, y):

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size =0.2, random_state=0)  # split data randomly into train and test split of 80/20% resp.

        return X_train, X_test, y_train, y_test    # return the split data sets
        
    def __vectorize(self, X, X_train, X_test):

        self.vect.fit(X)                                # fit vecotr object to balanced data test column as X
        X_train_tfidf =  self.vect.transform(X_train)   # transform the train data
        X_test_tfidf =  self.vect.transform(X_test)     # transform the test data

        return X_train_tfidf, X_test_tfidf              # return transformed test and train data for modeling
    
    def __encodeLabels(self, y_train, y_test):

        y_train_enc = self.encoder.fit_transform(y_train)  # encode the class data for train set
        y_test_enc = self.encoder.fit_transform(y_test)    # encode the class data for test set
        
        return y_train_enc, y_test_enc                     # return encoded class data for modeling

    def __createDB(self, X, y):

        conn = db.connect(os.getcwd() + "\\data.db")
        X.to_sql("text", conn, if_exists='replace')
        y.to_sql("category", conn, if_exists='replace')
        conn.close()


    def __setupClf(self):

        try:
            print("Start")
            balanced_data, X, y = self.__preprocess()
            print("one")
            X_train, X_test, y_train, y_test = self.__splitData(X, y)
            print("two")
            y_train_enc, y_test_enc = self.__encodeLabels(y_train, y_test)
            print("three")
            X_train_tfidf, X_test_tfidf = self.__vectorize(X, X_train, X_test)
            print("four")
            self.__createDB(X, y)
            print("Done")
            
            return [X_train_tfidf, y_train_enc, X_test_tfidf, y_test_enc]
            
        except Exception as e:
            print("Error: ", e)



    #public methods
    def get_model_info(self):

        return self.model_info    # get list of data for train/testing our models
    
    def input_vectorize(self, input_text):

        ip_text_tfidf = self.vect.transform(input_text)

        return ip_text_tfidf

    def decode_label(self, label):

        return self.encoder.inverse_transform(label)       

        
