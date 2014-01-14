

import numexpr as ne
#import matplotlib.pyplot as plt
from heapq import heappush
from heapq import nsmallest
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.random import shuffle
import random
from sklearn.decomposition import PCA, SparsePCA, RandomizedPCA
import hashlib

#sklearn.cluster.MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

def setX():
    loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=',')
    #amazon_old = loadData('anon-binclass.csv')
    amazon_new = loadData('../data/train_orig.csv')

    labels = amazon_new[1:,0]
    
    print "importing from dump!"
    orig_test = np.load('../data/X_test_all.dat')
    orig_train= np.load('../data/X_train_all.dat')
    num_features = orig_train.shape[1]
    
    X = orig_train#np.array([[100,100,100,100],[1,2,3,7],[101,108,87,50],[9,21,23,4],[90,98,12,155], [0,.3,0,-1],[5,5,5,5],[1,2,3,0]])
    Y = labels#np.array([0,1,0,1,0,1,1,1])
    return (X,Y)

def chooseNeighbor(neighbors_index,numNeighbors,to_be_removed):
    indices = neighbors_index[0]
    index_list = indices.tolist()
    index_list.remove(to_be_removed)
    index_list_size = len(index_list)
    if(index_list_size<numNeighbors):
        raise Exception('the num of neighbors if less than the number of points in the cluster')
    
    elif(index_list_size==numNeighbors):    
        return index_list
    #remaining_rows = index_list. 
    #create indices minus currRow
    else:
        listofselectedneighbors=[]
        for i in range(numNeighbors):
            selected_index = random.choice(index_list)
            listofselectedneighbors.append(selected_index)
            index_list.remove(selected_index)
        return listofselectedneighbors

def pca(X):
    '''reduce data'''
    pca = RandomizedPCA(n_components=3)
    Xreduced = pca.fit_transform(X, y=None)
    return Xreduced

def partitionSamples(X,Y):
    minority_rows=[]
    majority_rows=[]
    for i,row in enumerate(Y):
        if(1 == row):
            minority_rows.append(i)
        else:
            majority_rows.append(i)
    return (X[minority_rows],X[majority_rows])

def createSyntheticSamples(X,Y,nearestneigh,numNeighbors,majoritylabel,minoritylabel): 
    (Xminority,Xmajority) = partitionSamples(X,Y)
    numFeatures = Xminority.shape[1]
    Xreduced = pca(Xminority)
    numOrigMinority=len(Xminority)
    #reducedMinoritykmeans = KMeans(init='k-means++', max_iter=500,verbose=False,tol=1e-4,k=numCentroids, n_init=5, n_neighbors=3).fit(Xreduced)
    reducedNN = NearestNeighbors(nearestneigh, algorithm='auto')
    reducedNN.fit(Xreduced)
    #Xsyn=np.array([numOrigMinority,numNeighbors*numFeatures])
    trylist=[]
    #LOOPHERE  for EACH (minority) point...
    for i,row in enumerate(Xreduced):
        neighbor_index = reducedNN.kneighbors(row, return_distance=False) 
        closestPoints = Xminority[neighbor_index]
        #randomly choose one of the k nearest neighbors
        chosenNeighborsIndex = chooseNeighbor(neighbor_index,numNeighbors,i)
        chosenNeighbor = Xminority[chosenNeighborsIndex]
        #Calculate linear combination:        
        #Take te difference between the orig minority sample and its selected neighbor, where X[1,] is the orig point
        diff = Xminority[i,]-chosenNeighbor
        #Multiply this difference by a number between 0 and 1
        r = random.uniform(0,1)
        #Add it back to te orig minority vector and viola this is the synthetic sample
        syth_sample =Xminority[i,:]+r*diff
        syth_sample2 = syth_sample.tolist()
        trylist.append(syth_sample2)
    Xsyn=np.asarray(trylist).reshape(numNeighbors*numOrigMinority,numFeatures)
    maj_col=majoritylabel*np.ones([Xmajority.shape[0],1])
    min_col=minoritylabel*np.ones([Xsyn.shape[0],1])
    syth_Y = np.concatenate((maj_col,min_col),axis=0)
    syth_X = np.concatenate((Xmajority,Xsyn),axis=0)
    if(syth_X.shape[0]!=syth_Y.shape[0]):
        raise Exception("dim mismatch between features matrix and response matrix")
    return (syth_X, syth_Y)

if __name__ == '__main__':
    (myX,Y) = setX()
    reducedX = pca(myX)
    (Xsyn,Ysyn) = createSyntheticSamples(myX,Y,nearestneigh=4,numNeighbors=2,majoritylabel=1,minoritylabel=0) #k is the number of centroids, num_neighbors is the number of neighbors per each minority sample
    
    print Xsyn.shape
    print Ysyn.shape
    Xsyn.dump('../data/xsyn.dat')
    Ysyn.dump('../data/ysyn.dat')
    
    print(Xsyn)
    print(Ysyn)