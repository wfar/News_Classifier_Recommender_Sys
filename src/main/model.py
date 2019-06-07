#!/usr/bin/env python

''' model.py contains the ML classifier(s) and corresponding
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

# put nay changes to path here!!
PATH_TO_HP = os.getcwd() + "\\huffpo\\News_Category_Dataset_v2.json" 


__author_ = "{wfar}"
__version__ = "{v1.0}"


class Classifier:
    

    def __init__(self, model_type):
        self.type = model_type
        self.data = pd.read_json(PATH_TO_HP)
        
        
    def 
