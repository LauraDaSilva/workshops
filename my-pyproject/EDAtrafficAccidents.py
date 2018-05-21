# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:23:41 2018

File description: 
    - Visualization of Missing values
    - Correlation Matrix and plots
    - Univariate Visualization
    - Bivariate Visualization
    - Outliers
    - Imputations
    
@author: LauradaSilva
"""
import pandas as pd
import numpy as np
import impyute

# Library to visualize Missing values
import missingno as mn

# Library for statistical data visualization
import seaborn as sns

# Libraries for univariate and bivariate visualization
import matplotlib
import matplotlib.pyplot as plt
import ggplot
import plotly as py #interactive plot
# Library for data manipulation
from dfply import *

# Library for imputations
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

# Importing my own functions
import functions_data_description as fdd


# Getting the data
trfacc2016 = pd.read_csv('my-pyproject/Output/accidents2016.csv')

# Data types
# DateTime
trfacc2016["datetime"] = pd.to_datetime(trfacc2016["datetime"],
                                            format="%Y-%m-%d %H:%M")
# Categorical
trfacc2016["make"] = trfacc2016["make"].astype('category')
trfacc2016["model"] = trfacc2016["model"].astype('category')

# Full description
desctrfacc2016 = fdd.full_description(trfacc2016)

# Plotting Missing values
%matplotlib inline
#%matplotlib tk (separate window)
mn.matrix(trfacc2016)
# Visualizing correlations of missing values (pairs of columns)
mn.heatmap(trfacc2016)
# Visualizing correlation of missing values (more columns)
mn.dendrogram(trfacc2016)

# Imputations
# For continuous values -> Replace missing values with mean/median - impyute
# For discrete/categorical values -> classification model
# Let's impute the values of Speed_limit
# 1st type of imputation: Mean
# Mean imputation using Imputer from scikit-learn library
imp=Imputer(missing_values="NaN", strategy="mean" )
speedLimitMean =imp.fit_transform(trfacc2016[["Speed_limit"]]

# 2nd type of imputaiton: classifier
# Random forest model for multiclass classification
# Train and test split
train, test = trfacc2016[trfacc2016['Speed_limit'].isna() == False], \
              trfacc2016[trfacc2016['Speed_limit'].isna() == True]

# Delete those columns with NAs or with not influence in the prediction
X_train = train.drop(['datetime','month','Day_of_Week','model','make','hour','Speed_limit'],1)
y_train = train["Speed_limit"]
X_test = test.drop(['datetime','month','Day_of_Week','model','make','hour','Speed_limit'],1)
y_test = test["Speed_limit"]
#fd = fdd.full_description(X_train)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(X_train, y_train)
speedLimitPredicted = rf.predict(X_test)

# Replace the NAs with the predicted values for Speed_Limit
indexer = trfacc2016[trfacc2016['Speed_limit'].isna() == True].index
indexer = df[df.ids == encodedid].index
trfacc2016.loc[indexer, 'Speed_limit'] = speedLimitPredicted
fdd.full_description(trfacc2016)

# Correlation matrix and plots - only numeric values
# Compute the correlation matrix
corr = trfacc2016.corr()
# plot the heatmap
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(500, 50, as_cmap=True),
            #annot=True, fmt="f",
            square=True, linewidths=.5)

fdd.names(trfacc2016)


# When are the accidents happening?

# Histograms and Bar plots can be useful for this
# Day of the Week
plt.hist(trfacc2016['Day_of_Week'])
plt.title('Day of the Week')
plt.xlabel("Day of the week (1:Sunday - 7:Saturday)")
plt.ylabel("Frequency")

# Month
plt.hist(trfacc2016['month'].dropna())
plt.title('Month')
plt.xlabel("Month (1:January - 12:December)")
plt.ylabel("Frequency")

# Hour
plt.hist(trfacc2016['hour'].dropna())
plt.title('Hour')
plt.xlabel("Hour (0-23)")
plt.ylabel("Frequency")

# Where are the accidents happening?

# Country
plt.hist(trfacc2016['country'])
plt.title('Country')
plt.xlabel("Code (1:England, 2:Wales, 3:Scotland)")
plt.ylabel("Frequency")

# Area
# Different areas are assigned to different Police Forces
# So we can get more granularity
plt.hist(trfacc2016['Police_Force'])
plt.title('Police_Force')
plt.xlabel("Code")
plt.ylabel("Frequency")

num_bins = 65 
fig, ax = plt.subplots()
# the histogram of the data
n, bins, patches = ax.hist(trfacc2016['Police_Force'], num_bins, histtype='bar', rwidth=0.8)


# What type of vehicles are likely to have accidents? 

# Difficult to visualise given the amount of categories
sns.set(style="darkgrid")
ax = sns.countplot(x="make", data=trfacc2016)

# Aggregate and visualize it using a table
makeTable = (trfacc2016 >>
  group_by(X.make) >>
  summarize(makeCount = X.make.count()) #>>
  #arrange(X.makeCount) # It doesn't work
)

makeTable.sort_values('makeCount', ascending=False)


## Could you answer the following questions? 

# How are the weather conditions when the accidents are happening?


# Explore the same questions but with bivariate plots
# Relating accident severity and time, place, etc.


# Bivariate visualization
# Basic correlogram
sns.pairplot(trfacc2016.iloc[:,1:6])
# with regression
sns.pairplot(trfacc2016.iloc[:,2:6], kind="reg")
 # without regression
sns.pairplot(trfacc2016.iloc[:,1:6], kind="scatter")

 
# Outliers
# Boxplot
%matplotlib tk
trfacc2016.iloc[:,1:10].dropna().boxplot()










