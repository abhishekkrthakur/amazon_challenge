# Create test labels from all submissions...

import numpy as np
import os
import glob



loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=',')
 

fileCount = 0
path = '../sub/'
for infile in glob.glob( os.path.join(path, '*.csv') ):
    #print "current file is: " + infile
    fileCount += 1


temp = loadData('../sub/913.csv')
temp = temp[1:,1:]
p_all = np.empty((temp.shape[0], fileCount))

fileCount = 0
for infile in glob.glob( os.path.join(path, '*.csv') ):
	temp2 = loadData(infile)
	temp2 = temp2[1:,1:]
	p_all[:,fileCount] = np.reshape(temp2, (1,-1))
	fileCount += 1

pred = np.empty((p_all.shape[0],1))
for i in range(p_all.shape[0]):
	onePred = np.where(p_all[i] > 0.5)[0]
	zeroPred = np.where(p_all[i] <= 0.5)[0]

	if (len(onePred) > len(zeroPred)):
		pred[i] = 1
	else:
		pred[i] = 0


print pred
pred.dump('../data/predictedTest.dat')


