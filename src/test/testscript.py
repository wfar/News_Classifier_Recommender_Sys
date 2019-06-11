#!/usr/bin/env python

''' testscript.py contains the code to create all 3 models. Then using string input
    of news headlines, run the model to test prediction. New input
    can be added as text then passed through objects as shown in main().
    
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

    input_text = ["Meryl Streep was a scream in the 'Big Little Lies' Season 2 premiere"]   # entertainment
    input_text_2 = ["CFDA Fashion Awards 2019: Jennifer Lopez and Barbie among winners"]    # style
    input_text_3 = ["Trump teases additional deal that Mexican Foreign Secretary suggests doesn't exist"]  # politics

    tf1 = data_processor.input_vectorize(input_text)
    tf2 = data_processor.input_vectorize(input_text_2)
    tf3 = data_processor.input_vectorize(input_text_3)

    print("Input 1 pred using NB: " + data_processor.decode_label(NB_model.predict(tf1) ))
    print("Input 2 pred using NB: " + data_processor.decode_label(NB_model.predict(tf2) ))
    print("Input 3 pred using NB: " + data_processor.decode_label(NB_model.predict(tf3) ))
    print("Input 1 pred using LR: " + data_processor.decode_label(LR_model.predict(tf1) ))
    print("Input 2 pred using LR: " + data_processor.decode_label(LR_model.predict(tf2) ))
    print("Input 3 pred using LR: " + data_processor.decode_label(LR_model.predict(tf3) ))
    


if __name__ == "__main__":

    main()
