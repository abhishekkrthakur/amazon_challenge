import numpy as np
import scipy as sp
import sklearn.linear_model as lm
from sklearn.ensemble import RandomTreesEmbedding,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import metrics,preprocessing,cross_validation
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.lda import LDA
import utils
import SMOTE
import cv2
import pylab as pl
from rbm import BernoulliRBM
import matplotlib.cm as cm
from nolearn.dbn import DBN
import pywt
from scipy.optimize.optimize import fmin
import ADASYN
import Tomeklink
import rf
import data_io
import cyutils

doTest = 0

CreateSub = 0



loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=',')

amazon_new = loadData('../data/train_orig.csv')
labels = amazon_new[1:,0]
traindata = amazon_new[1:,1:] # 1-> 2

orig_lbl = labels
orig_lbl = np.reshape(orig_lbl,(-1,1))

print "importing from dump!"
orig_test = np.load('../data/testImage.dat')
orig_train= np.load('../data/trainImage.dat')
dbxtrain = np.load('../data/dbxtrain.dat')
dbxtest= np.load('../data/dbxtest.dat')
origTest = np.load('../data/X_test_all.dat')
origTrain = np.load('../data/X_train_all.dat')
num_features = orig_train.shape[1]
num_train = orig_train.shape[0]
print "import done.."

traindata = orig_train
testdata = orig_test



Xt = np.vstack((traindata,testdata))
# Extract good features using greedyFeatures.py
good_feat = [0, 1, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 52, 53]
Xt = Xt[:,good_feat]

Xt_orig1 = np.vstack((origTrain,origTest))
orig_good =   [0, 12, 15, 29, 44, 52, 65, 75, 83, 92, 94, 110, 115, 165, 170, 195, 202, 267, 272, 278, 323, 374, 405, 406, 407, 462, 473, 497, 524, 527, 528, 536, 539,540,541]
Xt_orig1 = Xt_orig1[:,orig_good]


print "normal hashing..."

lenc = preprocessing.LabelEncoder()

for i in range(5):
    lenc.fit(Xt_orig1)
    Xt_orig1 = lenc.transform(Xt_orig1)
    print i

print "Adding new features..."
xt_1 = Xt_orig1[:,1] * 0.002 + Xt_orig1[:,6]
xt_2 = Xt_orig1[:,4] * 2 + Xt_orig1[:,5] * 0.095
xt_3 = Xt_orig1[:,9] * 2 + Xt_orig1[:,10] * 0.055
xt_4 = Xt_orig1[:,6] * 0.2 + Xt_orig1[:,10] * 0.005
Xt_orig1 = np.column_stack((Xt_orig1,xt_1,xt_2,xt_3,xt_4))

#Xt_orig1 = utils.removeInfrequent2Thresh(Xt_orig1,1,2,9999)
#Xt_orig1.dump('../data/Xt_orig1_infreq_1_2.dat')

Xt_orig1 = np.load('../data/Xt_orig1_infreq_2_int32.dat')
Xt_orig12 = np.load('../data/Xt_orig1_infreq_1_2_newFeat.dat')
Xt_orig123 = np.load('../data/Xt_orig1_infreq_1_2_3.dat')

# traindata = Xt[:num_train]
# testdata = Xt[num_train:]

#np.savetxt("../data/train_replaced.csv", traindata, delimiter=",")
#np.savetxt("../data/test_replaced.csv", testdata, delimiter=",")


print "dump saved..!"

print "joining data..."
Xt = np.hstack((Xt,Xt_orig1))

Xt = Xt[:,[18, 19, 25, 42, 43, 46, 49, 52, 55, 59, 60, 61, 62, 63, 66, 67, 68, 69]]

traindata = Xt[:num_train]
testdata = Xt[num_train:]
print "perform one hot encoding .."
Xt,keymap = utils.OneHotEncoder(Xt)
traindata_sp = Xt[:num_train]
testdata_sp = Xt[num_train:]

#traindata.dump('../data/img_orig_train.dat')
#testdata.dump('../data/img_orig_test.dat')
#print "dumped!"

print "train data size:", traindata.shape
print "test data size:", testdata.shape


rforest = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=25, 
                               min_samples_split=1, min_samples_leaf=3, min_density=0.0000000000000001, max_features= "auto", bootstrap=True,
                               oob_score=True, n_jobs = -1,
                               random_state=None, verbose=1)

#rforest = ExtraTreesRegressor(n_estimators=100, verbose=1)

clf = DBN([traindata.shape[1], 20, 20, 2], momentum=0.9, learn_rates=0.3, learn_rate_minimums=0.01,learn_rate_decays=0.996,
            epochs=400,minibatch_size=100, epochs_pretrain=[100, 60, 60, 60],
            dropouts=[0.1, 0.5, 0.5, 0],momentum_pretrain=0.9, learn_rates_pretrain=[0.9, 0.9, 0.9, 0.9],
            verbose=1)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)



logres = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 
                            fit_intercept=True, intercept_scaling=1, class_weight="auto", 
                            random_state=None)

logres2 = lm.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.485994, 
                            fit_intercept=True, intercept_scaling=1, class_weight="auto", 
                            random_state=None)

rd = lm.Ridge(alpha = 64.0)

ab = AdaBoostClassifier(rforest,
                         algorithm="SAMME.R",
                         learning_rate=1.0,
                         n_estimators=10)


gb = GradientBoostingClassifier(loss='deviance', learning_rate=0.10000000000000001, n_estimators=10, subsample=1.0, 
                                min_samples_split=1, min_samples_leaf=3, max_depth=25,
                                random_state=None, max_features="auto", verbose=2)

ld = LDA()

gbr1 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=0.5, 
        min_samples_split=1, min_samples_leaf=3, max_depth=15, init=None, 
        random_state=None, max_features=None, alpha=0.9, verbose=2)

gbr2 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, 
        min_samples_split=1, min_samples_leaf=3, max_depth=15, init=None, 
        random_state=None, max_features="auto", alpha=0.9, verbose=2)

sv = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, shrinking=True,
tol=0.001, verbose=False)

# Select Greedy features from Xt_orig1
Xt_orig1 = Xt_orig1[:, [0, 1, 3, 5, 7, 9, 12, 13, 14, 16, 19, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36]]
Xt_orig12 = Xt_orig12[:, [0, 1, 3, 5, 7, 9, 12, 13, 14, 16, 19, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36]]
Xt_orig123 = Xt_orig123[:, [0, 1, 3, 5, 7, 9, 12, 13, 14, 16, 19, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37,38]]

print "encoding original data..."
Xt_orig, keymap = utils.OneHotEncoder(Xt_orig1)
train_orig = Xt_orig[:num_train]
test_orig = Xt_orig[num_train:]


print "encoding original data..."
Xt_orig12, keymap = utils.OneHotEncoder(Xt_orig12)
train_orig12 = Xt_orig12[:num_train]
test_orig12 = Xt_orig12[num_train:]

train_orig123nsp = Xt_orig1[:num_train]
test_orig123nsp = Xt_orig1[num_train:]

print "encoding original data..."
Xt_orig123, keymap = utils.OneHotEncoder(Xt_orig123)
train_orig123 = Xt_orig123[:num_train]
test_orig123 = Xt_orig123[num_train:]


#print "use ACTION for data generation..."
y, X = data_io.load_data_pd('../data/train_orig.csv', use_labels=True)
_, X_test = data_io.load_data_pd('../data/test_orig.csv', use_labels=False)

oldTest = loadData('../data/test_orig.csv')
oldTrain= loadData('../data/train_orig.csv')
oldTrain = oldTrain[1:,1:]
oldTest = oldTest[1:,1:]

old_all = np.vstack((X,X_test))
old_all = old_all.astype(np.int32, copy=False)
cyutils.removeInFrequent(old_all, 2, skipcols=-1)
#old_all = utils.removeInfrequent2Thresh(old_all,1,2,9999)

print "perform one hot encoding .."
Xt_old,keymap = utils.OneHotEncoder(old_all)
train_old_inf = Xt_old[:num_train]
test_old_inf = Xt_old[num_train:]

print "Cross Validation"
mean_auc = 0.0
n = 10  # repeat the CV procedure 10 times to get more precise results

nFeatures = X.shape[0]
niter = 10
SEED = 42
rnd = np.random.RandomState(SEED)
cct = 0
# xtrain = rf.getRFX(X)
# xtest = rf.getRFX_test(X_test)
# for j in range(xtest.shape[0]):
#     mini = np.min(xtest[j,:])
#     if mini == 0:
#         cct += 1

print "utils train test..."
util_train, util_test, util_train_nsp, util_test_nsp = utils.getTrainTest()

# blendTrain = np.empty((num_train, 3))
# blendTest  = np.zeros((X_test.shape[0],5))
# blendTestTemp1 =  np.empty((X_test.shape[0],10))
# blendTestTemp2 =  np.empty((X_test.shape[0],10))
# blendTestTemp3 =  np.empty((X_test.shape[0],10))
# blendTestTemp4 =  np.empty((X_test.shape[0],10))
# blendTestTemp5 =  np.empty((X_test.shape[0],10))
# blendTestTemp6 =  np.empty((X_test.shape[0],10))

n_folds = 10
verbose = True
shuffle = False

skf = list(StratifiedKFold(y, n_folds))

Xlr = train_orig
Xlr_test = test_orig
ylr = y.values

clfs = [#RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='gini'),
        #RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        #ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='gini'),
        #ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
#            GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        lm.LogisticRegression(C=1.0,random_state=None, tol=0.00001),                           #0
        lm.LogisticRegression(C=1.485994,random_state=None, class_weight='auto', tol=0.00001), #1
        lm.LogisticRegression(C=1.485994,random_state=None, class_weight='auto', tol=0.00001), #2
        lm.LogisticRegression(C=1.485994,random_state=None, class_weight='auto', tol=0.00001), #3
        lm.LogisticRegression(C=1.485994,random_state=None, class_weight='auto', tol=0.00001), #4
        lm.LogisticRegression(C=1.485994,random_state=None, class_weight='auto', tol=0.00001), #5
        #lm.LogisticRegression(C=1.485994,random_state=None, class_weight='auto', tol=0.00001), #6
        #GaussianNB()
        ]

print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((Xlr.shape[0], len(clfs)))
dataset_blend_test = np.zeros((Xlr_test.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((Xlr_test.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i            

        #j = 6
        xorig_train = train_orig[train]
        xorig_test = train_orig[test]

        xorig_train12 = train_orig12[train]
        xorig_test12 = train_orig12[test]

        xorig_train123 = train_orig123[train]
        xorig_test123 = train_orig123[test]

        xorig_train123nsp = train_orig123nsp[train]
        xorig_test123nsp = train_orig123nsp[test]

        img_orig_train = traindata[train]
        img_orig_test = traindata[test]

        img_orig_train_sp = traindata_sp[train]
        img_orig_test_sp = traindata_sp[test]

        xold_train = train_old_inf[train]
        xold_test = train_old_inf[test]

        utrain = util_train[train]
        utest = util_train[test]

        utrain_nsp = util_train_nsp[train]
        utest_nsp = util_train_nsp[test]

        if (j == 0):
            xtrain = xorig_train            
            xtest = xorig_test
            xsub = test_orig          
                             
            ytest = ylr[test]
            ytrain = ylr[train]            

            logres.fit(xtrain, ytrain)            
          
            dataset_blend_test_j[:, i] = logres.predict_proba(xsub)[:,1]
            y_submission = logres.predict_proba(xtest)[:,1]
            dataset_blend_train[test, j] = y_submission
            print y_submission

        if (j == 1):
            xtrain = img_orig_train            
            xtest = img_orig_test
            xsub = testdata        
                             
            ytest = ylr[test]
            ytrain = ylr[train]            

            gbr1.fit(xtrain, ytrain)            
          
            dataset_blend_test_j[:, i] = gbr1.predict(xsub)#[:,1]
            y_submission = gbr1.predict(xtest)#[:,1]
            dataset_blend_train[test, j] = y_submission
            print y_submission

        if (j == 2):
            xtrain = utrain            
            xtest = utest
            xsub = util_test        
                             
            ytest = ylr[test]
            ytrain = ylr[train]            

            logres2.fit(xtrain, ytrain)            
          
            dataset_blend_test_j[:, i] = logres2.predict_proba(xsub)[:,1]
            y_submission = logres2.predict_proba(xtest)[:,1]
            dataset_blend_train[test, j] = y_submission
            print y_submission

        if (j == 3):
            xtrain = xorig_train12           
            xtest = xorig_test12
            xsub = test_orig12          
                             
            ytest = ylr[test]
            ytrain = ylr[train]            

            logres.fit(xtrain, ytrain)            
          
            dataset_blend_test_j[:, i] = logres.predict_proba(xsub)[:,1]
            y_submission = logres.predict_proba(xtest)[:,1]
            dataset_blend_train[test, j] = y_submission
            print y_submission


        if (j == 4):
            xtrain = xorig_train123           
            xtest = xorig_test123
            xsub = test_orig123          
                             
            ytest = ylr[test]
            ytrain = ylr[train]            

            logres.fit(xtrain, ytrain)            
          
            dataset_blend_test_j[:, i] = logres.predict_proba(xsub)[:,1]
            y_submission = logres.predict_proba(xtest)[:,1]
            dataset_blend_train[test, j] = y_submission
            print y_submission

        if (j == 5):
            xtrain = img_orig_train_sp            
            xtest = img_orig_test_sp
            xsub = testdata_sp        
                             
            ytest = ylr[test]
            ytrain = ylr[train]            

            logres2.fit(xtrain, ytrain)            
          
            dataset_blend_test_j[:, i] = logres2.predict_proba(xsub)[:,1]
            y_submission = logres2.predict_proba(xtest)[:,1]
            dataset_blend_train[test, j] = y_submission
            print y_submission

        # if (j == 6):
        #     xtrain = img_orig_train_sp            
        #     xtest = img_orig_test_sp
        #     xsub = testdata_sp        
                             
        #     ytest = ylr[test]
        #     ytrain = ylr[train]            

        #     logres2.fit(xtrain, ytrain)            
          
        #     dataset_blend_test_j[:, i] = logres2.predict_proba(xsub)[:,1]
        #     y_submission = logres2.predict_proba(xtest)[:,1]
        #     #print y_submission, y_submission.shape,dataset_blend_train.shape
        #     dataset_blend_train[test, j] = y_submission
        #     print y_submission


        # if (j == 3):
        #     xtrain = xold_train            
        #     xtest = xold_test
        #     xsub = test_old_inf        
                             
        #     ytest = ylr[test]
        #     ytrain = ylr[train]            

        #     logres.fit(xtrain, ytrain)            
          
        #     #dataset_blend_test_j[:, i] = logres.predict_proba(xsub)[:,1]
        #     y_submission = logres.predict_proba(xtest)[:,1]
        #     dataset_blend_train[test, j] = y_submission
        #     print y_submission


        # if (j == 2):

        #     num_train = len(train)
        #     xtrain = X.ix[train]; ytrain = y.values[train]
        #     xtest = X.ix[test]; ytest = y.values[test]
        #     xtrain = rf.getRFX(xtrain)
        #     xtest = rf.getRFX_test(xtest)
        #     #xtrain = utils.get_numerical_features(xtrain, test=False)
        #     #xtest = utils.get_numerical_features(xtest, test=True)
        #     db_train = dbxtrain[train]
        #     db_test = dbxtrain[test]
        #     dbx = np.vstack((db_train,db_test))
        #     arr1 = dbx[:,0]
        #     #print dbx.shape
        #     for t in range(xtest.shape[0]):
        #         for k in range(xtest.shape[1]):
        #             if(xtest[t,k] == 0.0):
        #                 xtest[t,k] = min(np.std((xtest[t,:])) , np.std((xtest[:,k])))# + max(np.std((xtest[j,:])) , np.std((xtest[:,k]))))

        #     X_num = np.vstack((xtrain,xtest))
        #     newf1 = np.convolve(dbx[:,0],dbx[:,1],"same")

        #     X_num = np.column_stack((X_num, dbx[:,[0]]))
        #     xtrain = X_num[:num_train]
        #     xtest = X_num[num_train:]

        #     # xtrain = np.hstack((xtrain,db_train))
        #     # xtest = np.hstack((xtest, db_test))

        #     xorig_train = train_orig[train]
        #     xorig_test = train_orig[test]
        #     img_orig_train = traindata[train]
        #     img_orig_test = traindata[test]
        #     xold_train = train_old_inf[train]
        #     xold_test = train_old_inf[test]
        #     utrain = util_train[train]
        #     utest = util_train[test]


        #     #rforest.fit(xtrain, ytrain)
        #     #preds1 = logres.predict_proba(xtest)[:,1]
        #     #preds_1 = rforest.predict(xtest)
        #     #preds1 = preds_1
        #     #preds_11 = preds_1
            
        #     logres.fit(xorig_train, ytrain)
        #     preds2 = logres.predict_proba(xorig_test)[:,1]
        #     preds2_orig = preds2
        #     preds_2 = logres.predict(xorig_test)
        #     preds2p = preds2

        #     logres2.fit(utrain, ytrain)
        #     preds5 = logres2.predict_proba(utest)[:,1]
        #     #preds2_orig = preds2
        #     preds_5 = logres2.predict(utest)

        #     logres.fit(xold_train, ytrain)
        #     preds3 = logres.predict_proba(xold_test)[:,1]
        #     #preds2_orig = preds2
        #     preds_3 = logres.predict(xold_test)

        #     logres.fit(img_orig_train, ytrain)
        #     preds4 = logres.predict_proba(img_orig_test)[:,1]
        #     preds_4 = logres.predict(img_orig_test)

        #     rd.fit(utrain, ytrain)
        #     #preds1 = logres.predict_proba(xtest)[:,1]
        #     preds_6 = rd.predict(utest)
        #     preds6 = preds_6

        #     for p in range(len(preds2)):
        #         predictions = np.array([preds_2[p], preds_3[p], preds_4[p], preds_5[p]])
        #         predictions_proba = np.array([preds2[p], preds3[p], preds4[p], preds5[p]])
        #         nonzeroind = np.nonzero(predictions)[0]
        #         zeroind = np.where(predictions == 0)[0]
        #         lnonz = len(nonzeroind)
        #         # if (preds_1[p] == 1):
        #         #     lnonz += 1

        #         if (lnonz > 2):
        #             selectedPred = predictions_proba[nonzeroind]
        #             idxs = np.any(selectedPred != 1)
        #             selectedPred = selectedPred[idxs]
        #             preds2[p] = np.mean(selectedPred)

        #             #print np.mean(selectedPred)
        #         else:
        #             selectedPred = predictions_proba[zeroind]
        #             #idxs = np.any(selectedPred != 0)
        #             #selectedPred = selectedPred[idxs]
        #             preds2[p] = np.min(selectedPred)

        #         #print preds2[p], predictions_proba

        #     #preds2 = (preds2 - preds2.min())/(preds2.max() - preds2.min())
        #     dataset_blend_train[test, j] = preds2


        #     #y = labels
        #     #ytrain = y
        #     xtrain = rf.getRFX(X.ix[train])
        #     xsub = rf.getRFX_test(X_test)
        #     print xtrain.shape, xsub.shape
        #     dbx = np.vstack((db_train,dbxtest))
        #     t = 0
        #     k = 0

        #     for t in range(xsub.shape[0]):
        #         for k in range(xsub.shape[1]):
        #             if(xsub[t,k] == 0.0):
        #                 xsub[t,k] = min(np.std((xsub[t,:])) , np.std((xsub[:,k])))#/2.0

        #     X_num = np.vstack((xtrain,xsub))
        #     #newf1 = np.convolve(X_num[:,0],X_num[:,1],"same")

        #     X_num = np.column_stack((X_num,dbx[:,[0]]))
        #     xtrain = X_num[:num_train]
        #     xsub = X_num[num_train:]

        #     # xtrain = np.hstack((xtrain,db_train))
        #     # xtest = np.hstack((xtest, db_test))

        #     #xorig_train = train_orig[train]
        #     xorig_test = test_orig
        #     #img_orig_train = traindata
        #     img_orig_test = testdata

        #     #rforest.fit(xtrain, ytrain)
        #     #preds1 = rforest.predict_proba(xtest)[:,1]
        #     #preds_1 = rforest.predict(xsub)
        #     #preds1 = preds_1

        #     logres.fit(xorig_train, ytrain)
        #     preds2 = logres.predict_proba(xorig_test)[:,1]
        #     preds2_orig = preds2
        #     preds_2 = logres.predict(xorig_test)
        #     preds2p = preds2

        #     logres.fit(xold_train, ytrain)
        #     preds3 = logres.predict_proba(test_old_inf)[:,1]
        #     preds_3 = logres.predict(test_old_inf)

        #     logres.fit(img_orig_train, ytrain)
        #     preds4 = logres.predict_proba(img_orig_test)[:,1]
        #     preds_4 = logres.predict(img_orig_test)

        #     logres2.fit(utrain, ytrain)
        #     preds5 = logres2.predict_proba(util_test)[:,1]
        #     preds_5 = logres2.predict(util_test)

           
        #     for p in range(len(preds2)):
        #         predictions = np.array([preds_2[p], preds_3[p], preds_4[p], preds_5[p]])
        #         predictions_proba = np.array([preds2[p], preds3[p], preds4[p], preds5[p]])
        #         nonzeroind = np.nonzero(predictions)[0]
        #         zeroind = np.where(predictions == 0)[0]
        #         if (len(nonzeroind) > 2):
        #             selectedPred = predictions_proba[nonzeroind]
        #             idxs = np.any(selectedPred != 1)
        #             selectedPred = selectedPred[idxs]
        #             preds2[p] = np.mean(selectedPred)


        #             #print np.mean(selectedPred)
        #         else:
        #             selectedPred = predictions_proba[zeroind]
        #             idxs = np.any(selectedPred != 0)
        #             selectedPred = selectedPred[idxs]
        #             preds2[p] = np.min(selectedPred)


        #     #y_submission = preds2
        #     dataset_blend_test_j[:, i] = preds2
        #     #dataset_blend_train[test, j] = y_submission
        #     #print y_submission

    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

          
dataset = None; ytrue = None;
n_models = len(clfs)
x0 = np.ones((n_models, 1)) / n_models
def fopt_pred(pars, data):
    return np.dot(data, pars)
def fopt(pars):
    fpr, tpr, _ = metrics.roc_curve(ytrue, fopt_pred(pars, dataset))
    return -metrics.auc(fpr, tpr)

niter = 10
cv = cross_validation.ShuffleSplit(dataset_blend_train.shape[0], n_iter=niter, test_size=0.3, random_state=rnd)
mean_auc = 0.0; itr = 0       
for train, test in cv:
    xtrain = dataset_blend_train[train]
    xtest = dataset_blend_train[test]
    ytrain = ylr[train]
    ytest = ylr[test]        

    dataset = xtrain
    ytrue = ytrain
    xopt = fmin(fopt, x0)
    preds = fopt_pred(xopt, xtest)
    
    fpr, tpr, _ = metrics.roc_curve(ytest, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC (fold %d/%d): %f" % (itr + 1, niter, roc_auc)
    mean_auc += roc_auc ; itr += 1
print "Mean AUC: ", mean_auc/niter    

  
print "blending on full data..."    
def fopt_pred1(pars, data):
    return np.dot(data, pars)

def fopt1(pars):
    fpr, tpr, _ = metrics.roc_curve(ylr, fopt_pred1(pars, dataset_blend_train))
    return -metrics.auc(fpr, tpr)

dataset = dataset_blend_train
ytrue = ylr
xopt = fmin(fopt1, x0)
preds = fopt_pred1(xopt, dataset_blend_test)


filename = raw_input("Enter name for submission file: ")
utils.save_results(preds, "../sub/" + filename + ".csv")

print "Done..."

