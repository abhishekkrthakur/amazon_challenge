import numpy as np
import scipy as sp
import sklearn.linear_model as lm
from sklearn.ensemble import RandomTreesEmbedding,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn import metrics,preprocessing,cross_validation
import utils

loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=',')

amazon_new = loadData('../data/train_orig.csv')
labels = amazon_new[1:,0]
traindata = amazon_new[1:,1:] # 1-> 2

orig_lbl = labels
orig_lbl = np.reshape(orig_lbl,(-1,1))

print "importing from dump!"
orig_test = np.load('../data/X_test_all.dat')
orig_train= np.load('../data/X_train_all.dat')
num_features = orig_train.shape[1]
ortrain = orig_train
ortest = orig_test
print "import done.."

traindata = orig_train 
testdata = orig_test
print labels
print traindata.shape, labels.shape

Xt = traindata
print "Adding new features..."
xt_1 = Xt[:,1] * 0.002 + Xt[:,6]
xt_2 = Xt[:,4] * 2 + Xt[:,5] * 0.095
xt_3 = Xt[:,9] * 2 + Xt[:,10] * 0.055
xt_4 = Xt[:,6] * 0.2 + Xt[:,10] * 0.005
Xt = np.column_stack((Xt,xt_1,xt_2,xt_3,xt_4))
traindata = Xt

Xt = testdata
print "Adding new features..."
xt_1 = Xt[:,1] * 0.002 + Xt[:,6]
xt_2 = Xt[:,4] * 2 + Xt[:,5] * 0.095
xt_3 = Xt[:,9] * 2 + Xt[:,10] * 0.055
xt_4 = Xt[:,6] * 0.2 + Xt[:,10] * 0.005
Xt = np.column_stack((Xt,xt_1,xt_2,xt_3,xt_4))
testdata = Xt

# traindata = np.hstack((X,X_2,X_3,X_4))
print "selecting good features..."
good_1 =  [0, 11, 12, 53, 66, 67, 68, 71, 74, 75, 76, 82, 83, 92, 95, 105, 115, 127, 137, 139, 143, 165, 167, 171, 200, 203, 213, 224, 225, 226]
orig_good =   [0, 12, 15, 29, 44, 52, 65, 75, 83, 92, 94, 110, 115, 165, 170, 195, 202, 267, 272, 278, 323, 374, 405, 406, 407, 462, 473, 497, 524, 527, 528, 536, 539]

traindata = traindata[:,orig_good]
testdata = testdata[:,orig_good]
num_train = traindata.shape[0]
 
Xt = np.vstack((traindata,testdata))
print "zero mean..."
meandata = np.mean(Xt, axis = 0)
print meandata.shape
for i in range(Xt.shape[0]):
    Xt[i] -= meandata
    
traindata = Xt[:num_train]
testdata = Xt[num_train:]


print "normal hashing..."
lenc = preprocessing.LabelEncoder()
Xt = np.vstack((traindata,testdata))
for i in range(10):
    lenc.fit(Xt)
    Xt = lenc.transform(Xt)
    print i

traindata = Xt[:num_train]
testdata = Xt[num_train:]

print "Adding new features to train..."
xt_1 = traindata[:,1]  + traindata[:,6]
xt_2 = traindata[:,4] + traindata[:,5]
xt_3 = traindata[:,-9] + traindata[:,1] 
xt_4 = traindata[:,-1] + traindata[:,20] 
xt_5 = traindata[:,-2] + traindata[:,21] 
xt_6 = traindata[:,-3]**0.2 + traindata[:,22] 
xt_7 = traindata[:,10]**2  + traindata[:,11]**2
xt_8 = traindata[:,14] + traindata[:,15]**2
xt_9 = traindata[:,-19] + traindata[:,-11] 
xt_10 = traindata[:,-18] + traindata[:,-20] 
xt_11 = np.mean(traindata, axis = 1)
xt_12 = np.std(traindata, axis=1)
xt_13 = traindata[:,0]**2 + 0.5 *traindata[:,-3]
xt_14 = traindata[:,22]**2 + 0.5 *traindata[:,-23]
xt_15 = traindata[:,24]**2 + 0.5 *traindata[:,-24]
xt_16 = np.correlate(traindata[:,0],traindata[:,1],"same") 
print xt_16.shape

traindata2 = np.column_stack((xt_1,xt_2,xt_3,xt_4,xt_5,xt_6,xt_7,xt_8,xt_9,xt_10,xt_11,xt_12,xt_13,xt_14,xt_15,xt_16))
traindata2 = traindata2[:,[0, 5, 6, 7, 8, 12, 14]]
xt_16 = (np.sqrt(traindata[:,2] + traindata[:,1])) 

traindata2 = np.column_stack((traindata2,xt_16))

print "Adding new features to test..."
xt_1 = testdata[:,1]  + testdata[:,6]
xt_2 = testdata[:,4] + testdata[:,5]
xt_3 = testdata[:,-9] + testdata[:,1] 
xt_4 = testdata[:,-1] + testdata[:,20] 
xt_5 = testdata[:,-2] + testdata[:,21] 
xt_6 = testdata[:,-3]**0.2 + testdata[:,22] 
xt_7 = testdata[:,10]**2  + testdata[:,11]**2
xt_8 = testdata[:,14] + testdata[:,15]**2
xt_9 = testdata[:,-19] + testdata[:,-11] 
xt_10 = testdata[:,-18] + testdata[:,-20] 
xt_11 = np.mean(testdata, axis = 1)
xt_12 = np.std(testdata, axis=1)
xt_13 = testdata[:,0]**2 + 0.5 *testdata[:,-3]
xt_14 = testdata[:,22]**2 + 0.5 *testdata[:,-23]
xt_15 = testdata[:,24]**2 + 0.5 *testdata[:,-24]
xt_16 = np.correlate(testdata[:,0],testdata[:,1],"same") 

print xt_16.shape

testdata2 = np.column_stack((xt_1,xt_2,xt_3,xt_4,xt_5,xt_6,xt_7,xt_8,xt_9,xt_10,xt_11,xt_12,xt_13,xt_14,xt_15,xt_16))
testdata2 = testdata2[:,[0, 5, 6, 7, 8, 12, 14]]

xt_16 = (np.sqrt(testdata[:,2] + testdata[:,1])) 

testdata2 = np.column_stack((testdata2,xt_16))

print traindata2.shape

Xt = np.vstack((traindata,testdata))
#Xt = Xt[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38]]
## DEPRECATED!
# classifier3 = lm.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, 
#                               l1_ratio=.14999999999999999, fit_intercept=True, n_iter=50, shuffle=False, 
#                               verbose=0, epsilon=0.10000000000000001, n_jobs=-1, 
#                               random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, 
#                               class_weight=None, warm_start=False, rho=None, seed=None)
# #===============================================================================
# num_features = dbxtrain.shape[1]
# Xts = [OneHotEncoder(dbxtrain[:,[i]])[0] for i in range(num_features)]
#     
# print "Performing greedy feature selection..."
# score_hist = []
# N = 10
# good_features = set([])
# # Greedy feature selection loop
# while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
#     scores = []
#     for f in range(len(Xts)):
#         if f not in good_features:
#             feats = list(good_features) + [f]
#             Xtp = sparse.hstack([Xts[j] for j in feats]).tocsr()
#             score = cv_loop(Xtp, labels, classifier3, N)
#             scores.append((score, f))
#             print "Feature: %i Mean AUC: %f" % (f, score)
#     good_features.add(sorted(scores)[-1][1])
#     score_hist.append(sorted(scores)[-1])
#     print "Current features: %s" % sorted(list(good_features))
#     
# # Remove last added feature from good_features
# good_features.remove(score_hist[-1][1])
# good_features = sorted(list(good_features))
# print "Selected features %s" % good_features
#    
# xt2 = np.vstack((tt_1,tt_2))
#    
# xt2 = xt2[:,good_features]
# print xt2
# #rfe = RFE(estimator=classifier3, n_features_to_select=2, step=1,verbose = 3)
# #rfe.fit(tt_1, labels)
# #xt2 = rfe.transform(xt2)
# print xt2.shape
# Xt = np.column_stack((Xt,xt2))
#===============================================================================

#Xt = Xt.column_stack((Xt,ctX))

traindata = Xt[:num_train]
testdata = Xt[num_train:]


print traindata.shape
print "perform one hot encoding .."
Xt = np.vstack((traindata,testdata))
untraindata = Xt[:num_train]
untestdata = Xt[num_train:]
print "un-encoded data ready..."
Xt, keymap = utils.OneHotEncoder(Xt)
#print Xt.shape
traindata = Xt[:num_train]
testdata = Xt[num_train:]
print "encoded data ready..."
print traindata.shape
print testdata.shape



classifier1 = lm.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, 
                              l1_ratio=.14999999999999999, fit_intercept=True, n_iter=5000, shuffle=False, 
                              verbose=0, epsilon=0.10000000000000001, n_jobs=-1, 
                              random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, 
                              class_weight=None, warm_start=False, rho=None, seed=None)

classifier2 = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=20, 
                               min_samples_split=3, min_samples_leaf=25, min_density=0.0000000000000001, max_features="auto", bootstrap=True,
                               oob_score=True, n_jobs=-1,
                               random_state=None, verbose=2)

classifier3 = lm.LogisticRegression(C = 1.54)

print "Cross Validation"
X = traindata
y = labels
mean_auc = 0.0
n = 10  # repeat the CV procedure 10 times to get more precise results

#===============================================================================
for i in range(n):
    # for each iteration, randomly hold out 20% of the data as CV set
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
        X, y, test_size=.20, random_state=i*25)



    classifier3.fit(X_train, y_train)

    preds1 = classifier3.predict_proba(X_cv)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds1)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
    #print roc_auc
    mean_auc += roc_auc
  
print "Mean AUC: %f" % (mean_auc/n)
#===============================================================================


classifier3.fit(traindata, labels) 
preds1 = classifier3.predict_proba(testdata)[:,1]


filename = raw_input("Enter name for submission file: ")
utils.save_results(preds1, "../sub/" + filename + ".csv")

print "Done..."
