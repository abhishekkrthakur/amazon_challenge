'''
Created on Jul 7, 2013

'''

import numpy as np
import data_io
from sklearn import preprocessing, linear_model, ensemble
from sklearn import cross_validation
from sklearn.metrics import metrics
import pandas as pd


dicts = []

def getRFX(X):
    global dicts
    newx = []
    for c in range(1,X.shape[1]):
        
        #print"Numerical features for  ACTION, ", X.columns[c]        
        xdf = X.ix[:,[0,c]]
        xdf = xdf[xdf.ACTION > 0]
        xdf = pd.DataFrame(xdf.ix[:,1])
        grouped = xdf.groupby(xdf.columns[0], sort=False)
        groups1 = grouped.groups
        
        xdf = X.ix[:,[0,c]]
        xdf = xdf[xdf.ACTION < 1]
        xdf = pd.DataFrame(xdf.ix[:,1])
        grouped = xdf.groupby(xdf.columns[0], sort=False)
        groups0 = grouped.groups
        
        dicts.append((groups0, groups1))
        
        cfeature = []
        for r in range(X.shape[0]):
            key = X.values[r,c]
            t0 = groups0.get(key)
            t1 = groups1.get(key)
            nt0 = len(t0)*1.0 if t0 != None else 0.0
            nt1 = len(t1)*1.0 if t1 != None else 0.0
            v = nt1/(nt0+nt1)
            cfeature.append(v)
            
        newx.append(cfeature)
        
    
    newx = np.transpose(newx) 
    
    return newx   


def getRFX_test(Xtest):
    global dicts
    
    newxtest = []
    for c in range(1, Xtest.shape[1]):
        groups0, groups1 = dicts[c-1] 
        cfeature = []
        for r in range(Xtest.shape[0]):
            key = Xtest.values[r,c]
            t0 = groups0.get(key)
            t1 = groups1.get(key)
            nt0 = len(t0)*1.0 if t0 != None else 0.0
            nt1 = len(t1)*1.0 if t1 != None else 0.0
            v = nt1/(nt0+nt1) if nt0+nt1 > 0 else 0
            cfeature.append(v)
            
        newxtest.append(cfeature)
    
    newxtest = np.transpose(newxtest) 
    return newxtest
    

def doCV():
    SEED = 42
    rnd = np.random.RandomState(SEED)
    
    model_lr = linear_model.LogisticRegression(C=3)
    model_rf = ensemble.RandomForestClassifier(n_estimators=10, 
                                             min_samples_split=10, 
                                             compute_importances=False, 
                                             n_jobs=2, 
                                             random_state=rnd, 
                                             verbose=2)
    
    print "loading data for random forest..."
    y, X = data_io.load_data_pd('train_orig.csv', use_labels=True)
    _, X_test = data_io.load_data_pd('test_orig.csv', use_labels=False)
    
    xtrain = getRFX(X)
    xtest = getRFX_test(X_test)
    xtrain = xtrain[:,1:]
    xtest = xtest[:,1:]
    
    xtrain.dump('num_train.dat')
    xtest.dump('num_test.dat')
    print "dumped..!"
    print "loading data for logistic regression..."
    ysp, Xsp = data_io.load_data('train_orig.csv')
    y_testsp, X_testsp = data_io.load_data('test_orig.csv', use_labels=False)
    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    #print Xsp.shape, X_testsp.shape
    encoder.fit(np.vstack((Xsp, X_testsp)))
    Xsp = encoder.transform(Xsp)  # Returns a sparse matrix (see numpy.sparse)
    X_testsp = encoder.transform(X_testsp)
    
    

    print "starting cross validation..."
    nFeatures = X.shape[0]
    niter = 10
    cv = cross_validation.ShuffleSplit(nFeatures, n_iter=niter, test_size=0.2, random_state=rnd)
    mean_auc = 0.0; i = 0
    for train, test in cv:
        xtrain = X.ix[train]; ytrain = y[train]
        xtest = X.ix[test]; ytest = y[test]
        
        xtrain_sp = Xsp[train]
        xtest_sp = X_testsp[test]
        ytrainsp = ysp[train]
        
        xtrain = getRFX(xtrain)
        xtest = getRFX_test(xtest)
        xtrain = xtrain[:,1:]
        xtest = xtest[:,1:]
       
        print 'fitting random forest....'
        model_rf.fit(xtrain, ytrain) 
        preds_rf = model_rf.predict_proba(xtest)[:, 1]
        
        print 'fitting logistic regression...'
        model_lr.fit(xtrain_sp, ytrainsp)
        preds_lr = model_lr.predict_proba(xtest_sp)[:,1]
        
        preds = [np.mean(x) for x in zip(preds_rf, preds_lr)]
        
        fpr, tpr, _ = metrics.roc_curve(ytest, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, niter, roc_auc)
        mean_auc += roc_auc ; i += 1
    print "Mean AUC: ", mean_auc/niter

if __name__ == '__main__':
    doCV()
        