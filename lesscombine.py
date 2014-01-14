import numpy as np
import scipy as sp
import sklearn.linear_model as lm
from sklearn.ensemble import RandomTreesEmbedding,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics,preprocessing,cross_validation
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
import utils
import SMOTE
import cv2
import pylab as pl
from rbm import BernoulliRBM
import matplotlib.cm as cm
from nolearn.dbn import DBN
import pywt
import ADASYN
import Tomeklink
import rf
import data_io



loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=',')

amazon_new = loadData('../data/train_orig.csv')
labels = amazon_new[1:,0]
traindata = amazon_new[1:,1:] # 1-> 2


rforest = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=25, 
                               min_samples_split=1, min_samples_leaf=3, min_density=0.0000000000000001, max_features= "auto", bootstrap=True,
                               oob_score=True, n_jobs = -1,
                               random_state=None, verbose=1)

clf = DBN([traindata.shape[1], 20, 20, 2], momentum=0.9, learn_rates=0.3, learn_rate_minimums=0.01,learn_rate_decays=0.996,
            epochs=400,minibatch_size=100, epochs_pretrain=[100, 60, 60, 60],
            dropouts=[0.1, 0.5, 0.5, 0],momentum_pretrain=0.9, learn_rates_pretrain=[0.9, 0.9, 0.9, 0.9],
            verbose=1)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)



logres = lm.LogisticRegression(C = 1.485994)

rd = lm.Ridge(alpha = 64.0)

ab = AdaBoostClassifier(rforest,
                         algorithm="SAMME.R",
                         learning_rate=1.0,
                         n_estimators=10)


gb = GradientBoostingClassifier(loss='deviance', learning_rate=0.10000000000000001, n_estimators=100, subsample=1.0, 
                                min_samples_split=1, min_samples_leaf=4, max_depth=15,
                                random_state=None, max_features=None, verbose=2)

ld = LDA()


print "use ACTION for data generation..."
y, X = data_io.load_data_pd('../data/train_orig.csv', use_labels=True)
_, X_test = data_io.load_data_pd('../data/test_orig.csv', use_labels=False)

	
print "Cross Validation"
mean_auc = 0.0
n = 10  # repeat the CV procedure 10 times to get more precise results

nFeatures = X.shape[0]
niter = 10
SEED = 25
rnd = np.random.RandomState(SEED)


# newxtest = np.empty((cct,xtrain.shape[1]))
# newlabels = np.empty((1,cct))
# new_index = np.empty((1,cct))

# for j in range(xtest.shape[0]):
#     mini = np.min(xtest[j,:])
#     if mini == 0:
#         newxtest[j] = xtest[j,:] 


#===============================================================================
cv = cross_validation.ShuffleSplit(nFeatures, n_iter=niter, test_size=0.3, random_state=rnd)
mean_auc = 0.0; i = 0
for train, test in cv:
    print "======================================= CROSS VALIDATION LOOP: ", (i+1)
    num_train = len(train)
    xtrain = X.ix[train]; ytrain = y.values[train]
    xtest = X.ix[test]; ytest = y.values[test]
    xtrain = rf.getRFX(xtrain)
    xtest = rf.getRFX_test(xtest)

    cct = 0
    for j in range(xtest.shape[0]):
        mini = np.min(xtest[j,:])
        if mini != 0:
            cct += 1

    print "total nonzero rows: ", cct
    print ytest.shape

    newxtest = np.empty((cct,xtest.shape[1]))
    newlabels = np.empty((cct,1))
    #new_index = np.empty((cct))
    ytest = np.array(ytest)

    cnt = 0
    for j in range(xtest.shape[0]):
        mini = np.min(xtest[j,:])
        if mini != 0:
            newxtest[cnt] = xtest[j,] 
            newlabels[cnt] = ytest[j]
            cnt += 1
            #print cnt


    rforest.fit(xtrain, ytrain)
    preds1 = rforest.predict_proba(newxtest)[:,1]
    preds_1 = rforest.predict(newxtest)
    for j in range(len(preds1)):
        if preds1[j] == 1:
            preds1[j] = 0.99
        if preds1[j] == 0:
            preds1[j] == 0.01

    fpr, tpr, _ = metrics.roc_curve(newlabels, preds1)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC (fold %d/%d): %f" % (i + 1, niter, roc_auc)
    mean_auc += roc_auc ; i += 1
print "Mean AUC: ", mean_auc/niter



 