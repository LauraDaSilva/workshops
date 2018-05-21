# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:14:25 2017
@author: LauradaSilva

File description:
    - Dataset details (number of columns and rows)
    - Simple summary statistics
    - Analysis of Missing data
    - Analysis of Unique values
    - Data selection

"""

# Importing Python libraries

# Library providing high-performance, easy-to-use data structures
# and data analysis tools
import pandas as pd
# Library for scientific computing with Python.
import numpy as np
# Library that provides a way of using operating system dependent functionality.
import os as os
# Library for Date and Time format
import datetime as dt

# Importing my own libraries
import functions_data_description as fdd


# Setting the working directory
# Print the actual working directory
os.getcwd()
# Change the working directory
os.chdir('/Users/LauradaSilva/Documents/my-pyproject')
# List files in the specified directory
os.listdir('/Users/LauradaSilva/Documents/my-pyproject/Data')

# Getting the Accidents data and saving it in a DataFrame
Accidents = pd.read_csv("Data/Accidents_2016.csv")
# Show the first five values of each feature
print(Accidents.head(5))
# Dimension of your data set (num of observations (rows) and features (columns))
Accidents.shape
# Descriptive Statistics
accidentsSummary = Accidents.describe().transpose()

# Getting the Casualties data and saving it in a DataFrame
Casualties = pd.read_csv("Data/Casualties2016.csv")
Casualties.head(5)
Casualties.shape
casualtiesSummary = Casualties.describe().transpose()

# Getting the Model data and saving it in a DataFrame
Model = pd.read_csv("Data/MakeModel2016.csv")
Model.head(5)
Model.shape
modelSummary = Model.describe().transpose()

# Getting the Vehicles data and saving it in a DataFrame
Vehicles = pd.read_csv("Data/Vehicles2016.csv")
Vehicles.head(5)
Vehicles.shape
vehiclesSummary = Vehicles.describe().transpose()


# Describe features one by one
fdd.describe_features_one_by_one(Accidents)
fdd.describe_features_one_by_one(Casualties)
fdd.describe_features_one_by_one(Casualties)


# Get a full description of the Data Frames
descAccidents = fdd.full_description(Accidents)
descCasualties = fdd.full_description(Casualties)
descModel = fdd.full_description(Model)
descVehicles = fdd.full_description(Vehicles)
# Note that Vehicles dataset covers the same info that Model
# So, only use the first 3 datasets: Accidents, Casualties and Model

# Get the names of the features and the index
namesAccidents = fdd.names(Accidents)
namesCasualties = fdd.names(Casualties)
namesModel = fdd.names(Model)

# Feature Selection by intuition
# Accidents
accidents2016Sel = Accidents.iloc[:, [0, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 24, 25, 26]]

descAccidents2016Sel = fdd.full_description(accidents2016Sel)

# Casualties
casualties2016Sel = Casualties.iloc[:, [0, 4, 5, 7, 13]]

descCasualties2016Sel = fdd.full_description(casualties2016Sel)

# Model
model2016Sel = Model.iloc[:, [0, 3, 8, 15, 19, 22, 23]]

descModel2016Sel = fdd.full_description(model2016Sel)

# Joining the three datasets

casualtiesWithoutDuplicates = casualties2016Sel.drop_duplicates(subset="Accident_Index")
modelWithoutDuplicates = model2016Sel.drop_duplicates(subset="Accident_Index")

accidents2016final = pd.merge(pd.merge(accidents2016Sel,
                                       casualtiesWithoutDuplicates,
                                       on='Accident_Index'),
                              modelWithoutDuplicates,
                              on='Accident_Index')

descAccidents2016final = fdd.full_description(accidents2016final)

accidents2016final[pd.isnull(accidents2016final["Time"])]
accidents2016final[accidents2016final["Date"] == "2016-02-12"]

# Data Types

# New column datetime: Date + Time
accidents2016final["Date"] = pd.to_datetime(accidents2016final["Date"],
                                            format='%d/%m/%Y').astype("str")

accidents2016final["datetime"] = accidents2016final["Date"] + " " + accidents2016final['Time'].dropna().astype("str")

accidents2016final["datetime"] = pd.to_datetime(accidents2016final["datetime"],
                                            format="%Y-%m-%d %H:%M")

# Categorical
accidents2016final["make"] = accidents2016final["make"].astype('category')
accidents2016final["model"] = accidents2016final["model"].astype('category')

# Derived columns
accidents2016final["hour"] = accidents2016final["datetime"].dt.hour
accidents2016final["month"] = accidents2016final["datetime"].dt.month
accidents2016final["country"] = np.select([accidents2016final["Police_Force"] < 53,
                                           accidents2016final["Police_Force"] > 90], [1, 3], default=2)


finalAccidents2016 = accidents2016final.drop(['Date', 'Time', 'Accident_Index'], 1)
# where 1 is the axis number (0 for rows and 1 for columns.)

# Write dataset in a csv file
finalAccidents2016.to_csv("C:\\Users\\LauradaSilva\\Documents\\my-pyproject\\Output\\accidents2016.csv",
                          index=False, sep=',')

