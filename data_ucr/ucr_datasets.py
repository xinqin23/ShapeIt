#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exctracting one class (the first class) from ucr datasets
    
from  webpage : https://www.cs.ucr.edu/~eamonn/time_series_data/
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

path = 'data/UCR_TS_Archive_2015/'


#dataset_name = '50words'
#dataset_name = 'ECGFiveDays'
#dataset_name = 'Strawberry'
# dataset_name = 'Coffee'
# dataset_name = 'OliveOil'
# dataset_name = 'Wafer'
# dataset_name = 'Yoga'
# dataset_name = 'Worms'
# dataset_name = 'Earthquakes'
# dataset_name = 'BirdChicken'
# dataset_name = 'Computers'
# dataset_name = 'InsectWingbeatSound'
# dataset_name = 'Cricket_Y'
# dataset_name = 'MedicalImages'




## MEAT DATA    
# dataset_name = 'Meat'  
# set_data = 'ALL' # TRAIN and TEST set

## WINE DATA
dataset_name = 'Wine'
set_data = 'ALL' # TRAIN and TEST set

## FISH DATA
# dataset_name = 'Fish' #TRAIN DATA
# set_data = 'TRAIN' # TRAIN set only

#dataset_name = 'Car'
#dataset_name = 'Plane'
#dataset_name = 'ItalyPowerDemand'
#dataset_name = 'Herring'




data = pd.read_csv(f"{path}{dataset_name}/{dataset_name}_{set_data}", header = None)

tot_class_number = len(np.unique(data[0]))
    
 
for class_number in np.arange(1, tot_class_number +1):
    
    data_class = data.loc[data[0] == class_number] # time series classified as 'class_number'
    
    print(class_number, len(data_class))
    for index_row in range(len(data_class)):
        
         dt = (data_class.iloc[index_row])[1:]
         n = len(dt)
         plt.plot(np.arange(n), dt)
         
    plt.title(f"Dataset: {dataset_name}; Class number: {class_number}")
    plt.show()
    
    
class_number = 1

data_1 = data.loc[data[0] == class_number] # time series classified as '1'

for index_row in range(len(data_1)):
    
    dt = (data_1.iloc[index_row])[1:]
    t = np.arange(len(dt))
    
    dictionary = {'timestamp': t, 'value': dt}
    positive_example = pd.DataFrame(data = dictionary)
    positive_example.to_csv(f"data_ucr/{dataset_name}/{dataset_name}_{index_row}.csv", index = False)


















