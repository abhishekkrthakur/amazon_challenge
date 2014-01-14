'''
Created on Jul 7, 2013

'''

import numpy as np
import pandas as pd

def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open(filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open(filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data

def load_data_pd(filename, use_labels=True):
    data = pd.read_csv(filename)
    data = data.ix[:,0:-1]  #skip last column
    
    if(use_labels):
        labels = pd.read_csv(filename).ix[:,0]
    else:
        labels = np.zeros(data.shape[0])
    return labels, data
        