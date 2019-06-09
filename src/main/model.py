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
import pickle

#third party mods
import numpy as np
import pandas as pd
from sklearn import *

#local mods
import Classifier.py as CL
import DataProcessor.py as DP

# put any changes to path here!!
PATH_TO_HP = os.getcwd() + "\\huffpo\\News_Category_Dataset_v2.json"


class Model:

    def __init__(self):

        pass
