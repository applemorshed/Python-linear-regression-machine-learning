import numpy

# NOTE: This template makes use of Python classes. If
# you are not yet familiar with this concept, you can
# find a short introduction here:
# http://introtopython.org/classes.html


class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self, lam=0.0):

        self.lam = lam

    def fit(self, X, t):
        
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        # prepare a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        # compute weights (solve system)
        diag = self.lam * len(X) * numpy.identity(X.shape[1])
        a = numpy.dot(X.T, X) + diag
        b = numpy.dot(X.T, t)
        self.w = numpy.linalg.solve(a, b)

    def predict(self, X):
       
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
        X = numpy.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        # compute predictions
        predictions = numpy.dot(X, self.w)

        return predictions

    def fitband(self, X, t, sig):

        #M = numpy. mean(X, axis=0)
        M = numpy.subtract(X, t)

        N = (((M - X)**2) / (2 * ((sig)**2)))
        A = numpy.exp(-N)
        # print(A)
        # to creat a diogional matrix i use linweighreg.py from class lecture
        A = A[:, -1]
        # print(A)
        A = numpy.diag(A)
        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        k = numpy.dot(X.T, A)
        m = numpy.dot(k, X)
        b = numpy.dot(numpy.linalg.inv(m), X.T)
        n = numpy.dot(b, A)
        W1 = numpy.dot(n, t)
        self.w1 = W1

        # compute weights (solve system)

        # diag = self.lam * len(X) * numpy.identity(X.shape[1])
        # a = numpy.dot(X.T, X) + diag
        # b = numpy.dot(X.T, t)
        # self.w = numpy.linalg.solve(a,b)
    def predict_b(self, X):

        X = numpy.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        # compute predictions
        predictions = numpy.dot(X, self.w1)

        return predictions
