from .rbm import BernoulliRBM
import numpy

from dataConversions import indexToOneHot

def crossEntropy(p, l):
    """
    """
    nCases = l.size
    ce = 0

    for c in range(nCases):
        ce = ce - numpy.log2(p[c, l[c]])
    ce = ce / nCases

    return ce


class DRBM(BernoulliRBM):
    """Discriminative Restricted Boltzmann Machine

    Similar to BernoulliRBM, with an additional function to predict class
    labels. The training is still generative.
    """
    
    def __init__(self, nClass=2, n_components=256, learning_rate=0.01,
    batch_size=100, n_iter=10, verbose=False, random_state=None):
        """
        """
        super(DRBM, self).__init__(n_components, learning_rate, batch_size,
        n_iter, verbose, random_state)

        self.nClass = nClass

        # self.score = 100


    def fit(self, X, y=None):
        """
        Generative training of the DRBM.
        """

        yOneHot = indexToOneHot(y, self.nClass)

        Xy = numpy.concatenate((X, yOneHot), axis = 1)

        BernoulliRBM.fit(self, Xy)

    
    def score(self, X, y):
        """
        THERE'S STILL SOMETHING WRONG WITH THE PREDICTION PART OF THIS
        FUNCTION. BUT CHECK THE EVALUATION FUNCTION AS WELL IN ANY CASE.
        """

        nClass = self.nClass
        nPts, nDims = X.shape
        nHid = self.n_components
        W = self.components_ # of size nHid x nDims
        V = W[:, 0:nDims]
        U = W[:, nDims:nDims + nClass]
        a = self.intercept_hidden_
        d = self.intercept_visible_[nDims:nDims + nClass]

        Y = numpy.eye((nClass))

        Z1 = numpy.dot(V, numpy.transpose(X)) + numpy.transpose(numpy.tile(a, (nPts, 1)))
        Z2 = numpy.zeros((nHid, nPts, nClass))
        Z3 = numpy.zeros((1, nPts, nClass))

        for c in range(nClass):
            Z2[:, :, c] = numpy.dot(U, numpy.transpose(numpy.tile(Y[:, c], (nPts, 1))))
            Z3[:, :, c] = numpy.dot(numpy.tile(Y[:, c], (nPts, 1)), d)

        probs = numpy.zeros((nClass, nPts))
        pNorm = numpy.zeros((1, nPts))

        for c in range(nClass):
            Z4 = numpy.sum(numpy.log(1 + numpy.exp(Z1 + Z2[:, :, c])), axis=0)
            probs[c, :] = numpy.exp(Z3[:, :, c] + Z4)
            pNorm = pNorm + probs[c, :]

        probs = numpy.transpose(numpy.divide(probs, numpy.tile(pNorm, (nClass, 1))))
        cLabel = numpy.argmax(probs, axis=1)

        score = crossEntropy(probs, y)

        return score
