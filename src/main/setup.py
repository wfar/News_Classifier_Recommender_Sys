#!/usr/bin/env python

''' setup.py contains the source code to create both the dataprocessor singleton
    as well as the two ML models used for text classification. When ran, this file
    pickles and saves the dp as well as the models and creates a database of the balanced
    data. All are used on the web application.
    
'''

#builtin mods
import os
import sys
import time
import csv
import json
import pickle

#third party mods
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer

#local mods
import Classifier as CL
import DataProcessor as DP

# put any changes to path here!!
PATH_TO_HP = os.getcwd() + "\\huffpo\\News_Category_Dataset_v2.json" 


__author_ = "{wfar}"
__version__ = "{v1.0}"


def main():

    data_processor = DP.DataProcessor()
    info = data_processor.get_data_info()
    print(len(info))

    NB_model = CL.Classifier('NB', info[0], info[1], info[2], info[3] )
    LR_model = CL.Classifier('LR', info[0], info[1], info[2], info[3] )

    print("models created and dp created")
    
    pickle.dump(data_processor, open(os.getcwd() + "\\dataprocessor.pkl","wb"))
    pickle.dump(NB_model, open(os.getcwd() + "\\NBModel.pkl","wb"))
    pickle.dump(LR_model, open(os.getcwd() + "\\LRModel.pkl","wb"))

    print("objects pickled")

if __name__ == "__main__":

    main()

