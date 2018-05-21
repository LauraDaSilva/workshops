# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:39:04 2018
Description: Functions to describe the basic info of a DataFrame.
The actual functions are the following:
    - describe_features_one_by_one
    - full_description
    - names
@author: LauradaSilva
"""
import pandas as pd
import numpy as np

def describe_features_one_by_one(myDF):
    '''
    Description of each feature of a dataframe one by one.
    '''
    
    print("Dimension of this data frame", myDF.shape)
    print("----------------------------------------")
    var = "go"
    for feature in myDF:
        if var != "exit":
            print("----------------------------------------")
            print(myDF[feature].describe())
            print("----------------------------------------")
            var = input("Press any button to continue or write exit to finish \n")
        else:
            break


def full_description(myDF):
    '''
    Full description of a DataFrame.
    Includying: basic statistics + number of missing values + data types
    '''
    
    dfDescription = myDF.describe(include = "all").transpose()
    dfDescription["missingValues"] = myDF.isnull().sum()
    dfDescription["dataType"] = myDF.dtypes
    return dfDescription


def names(myDF):
    '''
    Get the names and indices of the features in the Data Frame
    '''
    
    index = list(range(0,len(myDF.columns),1))
    name = list(myDF.columns.values)
    nameDF = np.column_stack([index,name])#pd.DataFrame({index,name})
    return nameDF