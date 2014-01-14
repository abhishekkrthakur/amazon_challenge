import numpy
import numpy as np
from scipy import sparse
import cv2
import pandas as pd
from numpy import array
from itertools import combinations
from sklearn import  linear_model, preprocessing
from sklearn import cross_validation
from sklearn.metrics import metrics
from numpy import array
from scipy import sparse
import pandas as pd
from collections import Counter

SEED = 25

def getTrainTest(train='../data/train_edit.csv', test='../data/test_edit.csv', submit='logistic_pred.csv'):
#     print "Reading dataset..."
#     train_data = pd.read_csv(train)
#     test_data = pd.read_csv(test)
#     all_data = np.vstack((train_data.ix[:,1:], test_data.ix[:,1:]))

#     num_train = np.shape(train_data)[0]
    
    
#     lenc = preprocessing.LabelEncoder()
#     lenc.fit(all_data)
#     all_data = lenc.transform(all_data)
#     #X_test = scl.transform(X_test)
#     print all_data
#     # Transform data
#     print "Transforming data..."
# #  

#     dp = group_data(all_data, degree=2) 
#     dt = group_data(all_data, degree=3)
#     df = group_data(all_data, degree=4)

#     #y = array(train_data.ACTION)
#     X = all_data[:num_train]
#     X_2 = dp[:num_train]
#     X_3 = dt[:num_train]
#     X_4 = df[:num_train]
    

#     X_test = all_data[num_train:]
#     X_test_2 = dp[num_train:]
#     X_test_3 = dt[num_train:]
#     X_test_4 = df[num_train:]
# # 
#     X_train_all = np.hstack((X, X_2, X_3,X_4))
#     X_test_all = np.hstack((X_test, X_test_2, X_test_3,X_test_4))

    X_train_all = np.load('../data/X_train_all.dat')
    X_test_all = np.load('../data/X_test_all.dat')

    num_train = np.shape(X_train_all)[0]
    print X_train_all.shape
    
    good_features_t = [0, 8, 11, 12, 17, 26, 66, 67, 68, 71, 74, 75, 82, 83, 89, 92, 95, 105, 115, 150, 165, 203, 231, 235, 238, 239, 242, 245, 251, 263, 267, 273, 276, 279, 296, 300, 322, 324, 327, 359, 375, 380, 405, 406, 407, 408, 463, 469, 481, 485, 498, 528, 537, 539, 540, 541]
    
    Xt = np.vstack((X_train_all[:,good_features_t], X_test_all[:,good_features_t]))
    
    xt_1 = Xt[:,0] * 0.002 + Xt[:,1] 
    xt_2 = Xt[:,4] * 2 + Xt[:,5] * 0.095
    xt_3 = Xt[:,9] * 2 + Xt[:,10] * 0.055
    xt_4 = Xt[:,6] * 0.2 + Xt[:,10] * 0.005
    xt_5 = Xt[:,9] * 2 / Xt[:,10] * 0.02
    xt_6 = Xt[:,1] * 2 / Xt[:,10] * 0.030
    xt_7 = Xt[:,2] * 2 / Xt[:,9] * 0.04
    xt_8 = Xt[:,5] * 2 / Xt[:,6] * 0.05
    xt_9 = Xt[:,6] * 2 / Xt[:,2] 
    xt_10 = Xt[:,6] * 2 / Xt[:,4]
    xt_11 = xt_1/xt_10
    
    
    print xt_1.shape, Xt.shape
    xx = np.column_stack((Xt,xt_1,xt_2,xt_3,xt_4,xt_5,xt_6,xt_7,xt_8,xt_9,xt_10))
    dp1 = group_data(xx[:,[3,5,8,9,10,11]], degree=5) 
    dp2 = group_data(xx[:,[2,4,6,7,18,19]], degree=5)
    dp3 = group_data(xx[:,[15,17]], degree=2)
    dp4 = group_data(xx[:,[14,16]], degree=2)
    dp5 = group_data(xx[:,[15,17,19]], degree=3)
    dp6 = group_data(xx[:,[14,16,18]], degree=3)
    dp7 = dp1/dp2
    dp8 = dp3/dp4
    dp9 = dp5/dp6
    dp10 = dp7/dp8
    dp11 = dp8/dp9
    dp12 = dp9/dp7
    
    Xt = np.column_stack((Xt,xt_1,xt_2,xt_3,xt_4,xt_5,xt_6,xt_7,xt_8,xt_9,xt_10,xt_11,dp7,dp8,dp9,dp10,dp11,dp12,dp1,dp2,dp3,dp4,dp5,dp6))
    print "done stacking"

    print Xt.shape#,dp.shape

    feat = [0, 1, 2, 3, 5, 6, 7, 10, 15, 16, 17, 18, 19, 20, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 53, 54, 55, 58, 60, 61, 63, 94, 96, 97, 99, 101]
    
    Xt = Xt[:,feat]

    X_nsp = Xt[:num_train]
    X_test_nsp = Xt[num_train:]

    print "Encoding..."
    Xt, keymap = OneHotEncoder(Xt)  # Dont hide for log res.
    X = Xt[:num_train]
    X_test = Xt[num_train:]

    return X,X_test,X_nsp, X_test_nsp

def removeInfrequent(x, threshold, replacement):
    """
    Replace infrequent data with most infrequent data in a 
    2 dimensional numpy array (for every column)
    """
    print x.shape
    for i in range(x.shape[1]):
        temparr = x[:,i]
        print temparr, i
        temparr = temparr.astype(numpy.int32, copy=False)
        y = np.bincount(temparr)
        ii = np.nonzero(y)[0]
        freqCount = np.vstack((ii,y[ii])).T
        for j in range(freqCount.shape[0]):
            if (freqCount[j,1] < threshold):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement
        print x[:,i]

    return x

def FreqOneHotEncoder(data, keymap=None):
  if keymap is None:
    keymap = []
    for col in data.T:
      cc = Counter(list(col))
      uniques = set([elm for elm in list(col) if cc[elm]>1])
      keymap.append(dict((key, i) for i, key in enumerate(uniques)))
   

  total_pts = data.shape[0]
  outdat = []

  for i, col in enumerate(data.T):
    km = keymap[i]
    num_labels = len(km)
    spmat = sparse.lil_matrix((total_pts, num_labels))
    for j, val in enumerate(col):
      if val in km:
        spmat[j, km[val]] = 1
    outdat.append(spmat)

  outdat = sparse.hstack(outdat).tocsr()
  return outdat, keymap


def get_numerical_features(X, test=False):
    '''X contains all the columns '''
    global dicts    

    newx = []
    if(test):  #build numerical features for test set
        #add rate and fraction feature for all columns
        for c in range(1,X.shape[1]):  
            print "Test Numerical features for  ",X.columns[0], X.columns[c]   
            groups0, groups1 = dicts[c-1] 
            cfeature = [];  cfeature2 = []; 
            nfeats = X.shape[0]
            for r in range(nfeats):
                key = X.values[r,c]
                t0 = groups0.get(key)
                t1 = groups1.get(key)
                nt0 = len(t0)*1.0 if t0 != None else 0.0
                nt1 = len(t1)*1.0 if t1 != None else 0.0
                v = nt1/(nt0+nt1) if nt0+nt1 > 0 else 0  #rate of classifying to 1
                v2 = (nt0+nt1)/nfeats                    #fraction of examples containing this ID
                cfeature.append(v)  
                cfeature2.append(v2)                
            newx.append(cfeature)
            newx.append(cfeature2)        
        
    else:
        #rebuild dictionary of columns
        dicts = []
        #get features for train set   
        for c in range(1,X.shape[1]):        
            print"Train Numerical features for  ",X.columns[0], X.columns[c]        
            xdf = X.ix[:,[0,c]]
            xdf = xdf[xdf.values[:,0] > 0]
            xdf = pd.DataFrame(xdf.ix[:,1])
            grouped = xdf.groupby(xdf.columns[0], sort=False)
            groups1 = grouped.groups
            
            xdf = X.ix[:,[0,c]]
            xdf = xdf[xdf.values[:,0] < 1]
            xdf = pd.DataFrame(xdf.ix[:,1])
            grouped = xdf.groupby(xdf.columns[0], sort=False)
            groups0 = grouped.groups
            
            dicts.append((groups0, groups1))
            
            cfeature = []
            cfeature2 = []
            nfeats = X.shape[0]
            for r in range(nfeats):
                key = X.values[r,c]
                t0 = groups0.get(key)
                t1 = groups1.get(key)
                nt0 = len(t0)*1.0 if t0 != None else 0.0
                nt1 = len(t1)*1.0 if t1 != None else 0.0
                v = nt1/(nt0+nt1)
                v2 = (nt0+nt1)/nfeats
                cfeature.append(v)
                cfeature2.append(v2)                
            newx.append(cfeature)
            newx.append(cfeature2)
    
    newx = np.transpose(newx) 
    newx -= np.mean(newx, axis=0)
#    newx /= np.std(newx, axis=0)
    
    return newx


def removeInfrequent2Thresh(x, thresh1, thresh2, replacement):
    """
    Replace infrequent data with most infrequent data in a 
    2 dimensional numpy array (for every column)
    """
    print x.shape
    for i in range(x.shape[1]):
        temparr = x[:,i]
        print temparr, i
        temparr = temparr.astype(numpy.int32, copy=False)
        y = np.bincount(temparr)
        ii = np.nonzero(y)[0]
        freqCount = np.vstack((ii,y[ii])).T
        for j in range(freqCount.shape[0]):
            if (freqCount[j,1] == thresh1):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-1

            if (freqCount[j,1] == thresh2):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement

        print x[:,i]

    return x

def removeInfrequent2ThreshSepCol(x, thresh1, thresh2, replacement):
    """
    Replace infrequent data with most infrequent data in a 
    2 dimensional numpy array (for every column)
    """
    print x.shape
    for i in xrange(0,x.shape[1],2):
        temparr = x[:,i]
        print temparr, i
        temparr = temparr.astype(numpy.int32, copy=False)
        y = np.bincount(temparr)
        ii = np.nonzero(y)[0]
        freqCount = np.vstack((ii,y[ii])).T
        for j in xrange(0, freqCount.shape[0], 2):
            if (freqCount[j,1] == thresh1):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-1

            if (freqCount[j,1] == thresh2):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement

        print x[:,i]

    return x

def removeInfrequent3Thresh(x, thresh1, thresh2,thresh3, replacement):
    """
    Replace infrequent data with most infrequent data in a 
    2 dimensional numpy array (for every column)
    """
    print x.shape
    for i in range(x.shape[1]):
        temparr = x[:,i]
        print temparr, i
        temparr = temparr.astype(numpy.int32, copy=False)
        y = np.bincount(temparr)
        ii = np.nonzero(y)[0]
        freqCount = np.vstack((ii,y[ii])).T
        for j in range(freqCount.shape[0]):
            if (freqCount[j,1] == thresh1):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-1

            if (freqCount[j,1] == thresh2):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-2

            if (freqCount[j,1] == thresh3):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement
        print x[:,i]

    return x

def removeInfrequent4Thresh(x, thresh1, thresh2,thresh3,thresh4, replacement):
    """
    Replace infrequent data with most infrequent data in a 
    2 dimensional numpy array (for every column)
    """
    print x.shape
    for i in range(x.shape[1]):
        temparr = x[:,i]
        print temparr, i
        temparr = temparr.astype(numpy.int32, copy=False)
        y = np.bincount(temparr)
        ii = np.nonzero(y)[0]
        freqCount = np.vstack((ii,y[ii])).T
        for j in range(freqCount.shape[0]):
            if (freqCount[j,1] == thresh1):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-1

            if (freqCount[j,1] == thresh2):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-2

            if (freqCount[j,1] == thresh3):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement-3

            if (freqCount[j,1] == thresh4):
                a = np.where(x[:,i] == freqCount[j,0])
                x[:,i][a] = replacement
        print x[:,i]

    return x

def removeInfrequent_ColMin(x, threshold):
    """
    Replace infrequent data with most infrequent data in a 
    2 dimensional numpy array (for every column)
    """
    print x.shape
    for i in range(x.shape[1]):
        temparr = x[:,i]
        print temparr, i
        temparr = temparr.astype(numpy.int32, copy=False)
        y = np.bincount(temparr)
        ii = np.nonzero(y)[0]
        freqCount = np.vstack((ii,y[ii])).T
        for j in range(freqCount.shape[0]):
            if (freqCount[j,1] < threshold):
                a = np.where(x[:,i] == freqCount[j,0])
                replacement_index = np.unravel_index(freqCount[:,1].argmin(), freqCount[:,1].shape[0])
                replacement = freqCount[replacement_index,0]
                x[:,i][a] = replacement
        print x[:,i]

    return x

def logsig(x):
    return 1 / (1 + numpy.exp(-x))

def reconstruct(r, v):
    """
    RECONSTRUCT   Reconstruct the input vector to the RBM.
        
        This function takes as input a feature vector of the same form that
        was used to train the RBM, clamps it to the visible units of the
        RBM and uses one 'up-down' pass to reconstruct the input. I think
        this can also  be viewed as drawing one sample from the
        distribution of the units in the visible layer.
    """

    # Perform one up-down step to sample from the joint distribution
    p_hv0 = logsig(numpy.dot(v, numpy.transpose(r.components_)) + r.intercept_hidden_)
    s_h0 = (p_hv0 > numpy.random.rand(1, r.intercept_hidden_.size)).astype(int)
    neg_data = logsig(numpy.dot(s_h0, r.components_) + r.intercept_visible_)

    return neg_data

def surf_freak_detect(image,hessianThreshold):
    surfDetector = cv2.SURF(hessianThreshold)
    surfDetector = cv2.GridAdaptedFeatureDetector(surfDetector,50)
    keypoints = surfDetector.detect(image,None)
    freakExtractor = cv2.DescriptorExtractor_create('FREAK')
    keypoints,descriptors = freakExtractor.compute(image,keypoints)
    del freakExtractor
    return keypoints,descriptors

def fopt_pred(pars, data):
    return np.dot(data, pars)



def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))
            
def SMOTE(T, N, k):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """    
    n_minority_samples, n_features = T.shape
    
    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    N = N/100
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    
    #Calculate synthetic samples
    for i in xrange(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in xrange(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]
            print S
    
    return S

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

def CFAR_features(data,label):
    n = data.shape[0]
    c_p = 0
    c_n = 0
    all, bin = count_unique(data)
    newdata = np.empty((len(data),1))
    
    for i in range(n):
        if (label[i] == 1):
            c_p += 1
        else:
            c_n += 1
    
    tempdata_p = np.empty((c_p,1))
    tempdata_n = np.empty((c_n,1))
    j = 0
    k = 0
    
    for i in range(n):
        if (label[i] == 1):
            tempdata_p[j] = data[i]
            j += 1 
        if (label[i] == 0):
            tempdata_n[k] = data[i]
            k += 1 
    
    all_p, bin_p = count_unique(tempdata_p)
    all_n, bin_n = count_unique(tempdata_n)
    
    m = data.shape[0]
    
    for i in range(m):
        temp = data[i]
        if (label[i] == 1):
            ctn = bin_p[np.where(all_p == temp)]
            ctd = bin[np.where(all == temp)]
            newdata[i] = ctn/ctd 
        if (label[i] == 0):
            ctn = bin_n[np.where(all_n == temp)]
            ctd = bin[np.where(all == temp)]
            newdata[i] = ctn/ctd 
            
    return newdata

def polynomial_features(x, order):
    x = np.asarray(x).T[np.newaxis]
    n = x.shape[1]
    power_matrix = np.tile(np.arange(order + 1), (n, 1)).T[..., np.newaxis]
    X = np.power(x, power_matrix)
    I = np.indices((order + 1, ) * n).reshape((n, (order + 1) ** n)).T
    F = np.product(np.diagonal(X[I], 0, 1, 2), axis=2)
    return F.T

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def group_data2(data, degree=3, hashfn=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    _,n = data.shape
    
    for indicies in combinations(range(n), degree):
        if(degree==2 and ((set(indicies) == set([2,3]) or set(indicies)==set([5,6])))):  #skip rollup1-rollup2 and role_family-role_title combinations
            continue
        else:
            new_data.append([hash(tuple(v))& 0x7FFFFFF for v in data[:,indicies]])
            
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap
 

def smooth(x,window_len=10,window='hanning'):
    #print "in smooth method"
    #print x, window_len, window
    s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='same')
    dy = s.copy()
    dy[window_len-1:-window_len+1] = s[window_len-1:-window_len+1] - y[window_len-1:-window_len+1]
    return y[window_len-1:-window_len+1], dy[window_len-1:-window_len+1]