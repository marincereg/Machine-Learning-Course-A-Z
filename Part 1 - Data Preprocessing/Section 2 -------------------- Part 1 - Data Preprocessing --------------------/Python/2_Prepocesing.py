# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:14:11 2022

@author: marin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
"""

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
x = df.iloc[:,1]
print (x)
                  