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


__author_ = "{wfar}"
__version__ = "{v1.0}"


def main():

    data = DP.DataProcessor()
    data.run()

    
    



if __name__ == "__main__":

    main()

