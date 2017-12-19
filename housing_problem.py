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
#print miss

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


#print corr['SalePrice'].sort_values(ascending=False)[:28]

#see catorical data description
#print cat_data.describe()


#ANOVA defination 
#cat = [f for f in train.columns if train.dtypes[f] == 'object']
#
#def anova(frame):
#    anv = pd.DataFrame()
#    anv['features'] = cat
#    pvals = []
#    for c in cat:
#           samples = []
#           for cls in frame[c].unique():
#                  s = frame[frame[c] == cls]['SalePrice'].values
#                  samples.append(s)
#           pval = stats.f_oneway(*samples)[1]
#           pvals.append(pval)
#    anv['pval'] = pvals
#    return anv.sort_values('pval')
#
#cat_data['SalePrice'] = train.SalePrice.values
#k = anova(cat_data) 
#k['disparity'] = np.log(1./k['pval'].values) 
#sns.barplot(data=k, x = 'features', y='disparity') 
#plt.xticks(rotation=90) 
#plt 

#create numeric plots
#num = [f for f in train.columns if train.dtypes[f] != 'object']
#num.remove('Id')
#nd = pd.melt(train, value_vars = num)
#n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
#n1 = n1.map(sns.distplot, 'value')
#n1

#For categorial data
#def boxplot(x,y,**kwargs):
#            sns.boxplot(x=x,y=y)
#            x = plt.xticks(rotation=90)
#
#cat = [f for f in train.columns if train.dtypes[f] == 'object']
#
#p = pd.melt(train, id_vars='SalePrice', value_vars=cat)
#g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
#g = g.map(boxplot, 'value','SalePrice')
#g


# filling missing data
print stats.mode(test['GarageQual']).mode
print np.nanmedian(test['GarageYrBlt'])

#removing outliers
train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)
train.shape

#imputing using mode
test.loc[666, 'GarageQual'] = "TA" #stats.mode(test['GarageQual']).mode
test.loc[666, 'GarageCond'] = "TA" #stats.mode(test['GarageCond']).mode
test.loc[666, 'GarageFinish'] = "Unf" #stats.mode(test['GarageFinish']).mode
test.loc[666, 'GarageYrBlt'] = "1980" #np.nanmedian(test['GarageYrBlt'])

#mark as missing
test.loc[1116, 'GarageType'] = np.nan

#importing function
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def factorize(data, var, fill_na = None):
      if fill_na is not None:
            data[var].fillna(fill_na, inplace=True)
      le.fit(data[var])
      data[var] = le.transform(data[var])
      return data

#combine the data set
alldata = train.append(test)
alldata.shape

#impute lotfrontage by median of neighborhood
lot_frontage_by_neighborhood = train['LotFrontage'].groupby(train['Neighborhood'])

for key, group in lot_frontage_by_neighborhood:
                idx = (alldata['Neighborhood'] == key) & (alldata['LotFrontage'].isnull())
                alldata.loc[idx, 'LotFrontage'] = group.median()

#imputing missing values
alldata["MasVnrArea"].fillna(0, inplace=True)
alldata["BsmtFinSF1"].fillna(0, inplace=True)
alldata["BsmtFinSF2"].fillna(0, inplace=True)
alldata["BsmtUnfSF"].fillna(0, inplace=True)
alldata["TotalBsmtSF"].fillna(0, inplace=True)
alldata["GarageArea"].fillna(0, inplace=True)
alldata["BsmtFullBath"].fillna(0, inplace=True)
alldata["BsmtHalfBath"].fillna(0, inplace=True)
alldata["GarageCars"].fillna(0, inplace=True)
alldata["GarageYrBlt"].fillna(0.0, inplace=True)
alldata["PoolArea"].fillna(0, inplace=True)

qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])

for i in name:
     alldata[i] = alldata[i].map(qual_dict).astype(int)

alldata["BsmtExposure"] = alldata["BsmtExposure"].map({np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(bsmt_fin_dict).astype(int)
alldata["Functional"] = alldata["Functional"].map({np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

alldata["GarageFinish"] = alldata["GarageFinish"].map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)
alldata["Fence"] = alldata["Fence"].map({np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

#encoding data
alldata["CentralAir"] = (alldata["CentralAir"] == "Y") * 1.0
varst = np.array(['MSSubClass','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Foundation','SaleCondition'])

for x in varst:
         factorize(alldata, x)

#encode variables and impute missing values
alldata = factorize(alldata, "MSZoning", "RL")
alldata = factorize(alldata, "Exterior1st", "Other")
alldata = factorize(alldata, "Exterior2nd", "Other")
alldata = factorize(alldata, "MasVnrType", "None")
alldata = factorize(alldata, "SaleType", "Oth")
    
