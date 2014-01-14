#!/bin/python

from rbm import BernoulliRBM

import scipy
import scipy.io

import numpy

import matplotlib
from matplotlib import pyplot

import sklearn
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

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


def main():
    # Load digits data from sklearn
    digits = datasets.load_digits()
    features = digits.data
    (n_cases, n_dims) = features.shape

	# Binarize digits
    gray_max = numpy.max(features)
    gray_min = numpy.min(features)
    th = (gray_max - gray_min) * 0.65
    features_bin = (features > th).astype(int)

    # Initialize RBM structure and training parameters
    n_components = 256
    learning_rate = 0.1
    batch_size = 100
    n_iter = 100
    verbose = True

	# Declare RBM instance and generatively train it with the input data
    r = BernoulliRBM(n_components, learning_rate, batch_size, n_iter,
    verbose, None)
    r.fit(features_bin)

    idx = numpy.random.random_integers(0, n_cases -1);
    inp = features_bin[idx, :]
    im_inp = numpy.reshape(inp, (8, 8))
    	
    # Original digit image
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.imshow(im_inp)
    matplotlib.pyplot.gray()
    matplotlib.pyplot.title('Input Image')
    matplotlib.pyplot.show()
    
    print features_bin
	
    # Reconstruct input digit image feature
    rec = reconstruct(r, inp)
    im_rec = numpy.reshape(rec, (8, 8))

    # Reconstructed digit image
    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.imshow(im_rec)
    matplotlib.pyplot.gray()
    matplotlib.pyplot.title('Reconstructed Image')
    matplotlib.pyplot.show()
    
    print rec
	
	

if __name__ == "__main__":
	main()
