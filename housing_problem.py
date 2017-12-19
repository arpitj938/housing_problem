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

#no of  rows 1460 no of col 81 
print train.shape[0]

#no of col having empty values 

print train.columns[train.isnull().any()]

#missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
print miss

miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#plot the missing value count
#sns.set(style="whitegrid", color_codes=True)
#sns.barplot(x = 'Name', y = 'count', data=miss)
#plt.xticks(rotation = 90)
#sns.distplot(train['SalePrice'])

#skewness of target variable 
target = np.log(train['SalePrice'])
print ('Skewness is', target.skew())
#sns.distplot(target)

#separate variables into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))

del numeric_data['Id']

#correlation plot
corr = numeric_data.corr()
#sns.heatmap(corr)


print corr['SalePrice'].sort_values(ascending=False)[:28]



