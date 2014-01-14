'''
#===============================================================================
# call in unix terminal in current directory: python setup.py build_ext --inplace
#===============================================================================

@author: bhanu
'''

from __future__ import division
import numpy as np
cimport numpy as np
import cython
import pandas as pd
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
# We now need to fix a datatype for our arrays. I've used the variable
# ITYPE for this, which is assigned to the usual NumPy runtime
# type info object.
ITYPE = np.int32
# "ctypedef" assigns a corresponding compile-time type to ITYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int32_t ITYPE_t
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t

@cython.boundscheck(False)
def removeInFrequent(np.ndarray[ITYPE_t, ndim=2] data not None, 
                     int threshold, 
                     int skipcols=-1
                     ):
    datafr = pd.DataFrame(data[:,skipcols+1:]) #do not replace resource column
    dictionary = None
    
    cdef unsigned int col, nCols, r, category, nKeys
    cdef np.ndarray[ITYPE_t, ndim=1] countlist
    cdef np.ndarray[ITYPE_t, ndim=1] keys
    
    nCols = datafr.shape[1]
    for col in range(nCols):
        print "replacing infrequent categories, column: ", col
        df = pd.DataFrame(datafr.ix[:,col])
        grouped = df.groupby(df.columns[0], sort=False)
        dictionary = grouped.groups
        keys = np.array(dictionary.keys())
        nKeys = keys.shape[0]
        for i in range(nKeys):
#        for category, countlist in dictionary.items():
            category = keys[i]
            countlist = np.array(dictionary.get(category), dtype='int32')
            if(countlist.shape[0] < threshold):
                for r in range(data.shape[0]):
                    if(data[r,col] == category):
                        data[r,col] = 9999
        
        
#    #replace all infrequent cats with one cat
#    for c, ic in enumerate(infrequent_cats):
#        print "replacing infrequent categories, column: ", c," out of ",len(infrequent_cats), " columns..." 
#        for cat in ic:
#            for r in range(data.shape[0]):
#                if(datafr.values[r,c] == cat):
#                    datafr.values[r,c] = 9999