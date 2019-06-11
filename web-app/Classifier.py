#!/usr/bin/env python

''' Classifier.py contains the ML classifier(s) and corresponding
    methods used for text classification.
    
'''

#builtin mods
import os
import sys
import time
import csv
import json

#third party mods
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# put nay changes to path here!!
PATH_TO_HP = os.getcwd() + "\\huffpo\\News_Category_Dataset_v2.json" 


__author_ = "{wfar}"
__version__ = "{v1.0}"


class Classifier:
    
    # Constructor
    def __init__(self, model_type, X_train, y_train, X_test, y_test):
        self.type = model_type
        self.acc = None
        
        if self.type == 'NB':
            self.model = MultinomialNB(alpha = 0.05)
        elif self.type == 'LR':
            self.model = linear_model.LogisticRegression(multi_class='auto')
        else:
            self.model = None
            return "Invalid Model type"

        self.__trainModel(X_train, y_train)
        self.__testModel(X_test, y_test)

    #private methods    
    def __trainModel(self, X_train, y_train):

        self.model.fit(X_train, y_train)
        

    def __testModel(self, X_test, y_test):

        predictions = self.model.predict(X_test)
        self.acc = accuracy_score(y_test, predictions)

    #public methods
    def get_model_accuracy(self):

        return self.acc

    def get_model_type(self):

        return self.type

    def predict(self, input_data):

        prediction = self.model.predict(input_data)
        return prediction


        
        
        
