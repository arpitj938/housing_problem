# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:21:28 2017

@author: arpit
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print train.head()
print test.head()