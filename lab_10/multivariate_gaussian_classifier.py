import numpy as np
import scipy

class MultivariateGaussianClassifier:
    """ Gaussian Classifier for multivariate data
        The classifier is based on the Gaussian pdf
        The classifier assumes that the data is normally distributed
        The classifier assumes that the covariance matrix is the same for all classes

        How to use:
        1. Create an instance of the class
        2. Call the fit method to train the model: fit(X, y)
        3. Call the log_gau_pdf method to compute the log of the Gaussian pdf: log_gau_pdf(X, mu[i], sigma[i])
        4. Call the predict method to compute the accuracy: predict(log_score, y)
    """
    def __init__(self, prior = 0.5):
        self.mu = None
        self.sigma = None
        self.prior = prior
    
    @staticmethod
    def log_gau_pdf(X, mu, sigma):
        """ Compute the log of the Gaussian pdf
            X: n_features, n_samples
            mu: class mean
            sigma: class covariance matrix
        """
        
        if X.shape[0] < X.shape[1]:
            X = X.T
        
        _, d = X.shape # n: samples, d: features,
        X_c = X - mu.reshape(1, -1)
        inv_sigma = np.linalg.inv(sigma)
        sign, log_det_sigma = np.linalg.slogdet(sigma)
        det_sign = sign * log_det_sigma
        quad_form = np.sum(X_c @ inv_sigma * X_c, axis=1)
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + det_sign + quad_form)
        
        return log_pdf
    
    def fit(self, X, y):
        self.mu = [np.mean(X[:, y == i], axis=1).reshape(-1, 1) for i in np.unique(y)]
        self.sigma = [np.dot(X[:, y == i] - self.mu[i], (X[:, y == i] - self.mu[i]).T) / X[:, y == i].shape[1] for i in np.unique(y)]
    
    def predict(self, log_score, y):
        log_score = log_score + np.log(self.prior)
        margin_ls = scipy.special.logsumexp(log_score, axis=0)
        posterior_ls = log_score - margin_ls
        posterior_s = np.exp(posterior_ls)
        acc = np.argmax(posterior_s, axis=0) == y
        accuracy = np.sum(acc) / len(y)
        return accuracy
    
    def set_prior(self, prior):
        self.prior = prior
        
    def get_prior(self):
        return self.prior

    
class NaiveBayesClassifier(MultivariateGaussianClassifier):
    """ Naive Bayes Classifier for multivariate data 
        The classifier is based on the Gaussian pdf
        The classifier assumes that the data is normally distributed
        The classifier assumes that the covariance matrix is diagonal

        How to use:
        1. Create an instance of the class
        2. Call the fit method to train the model: fit(X, y)
        3. Call the log_gau_pdf method to compute the log of the Gaussian pdf: log_gau_pdf(X, mu[i], sigma[i])
        4. Call the predict method to compute the accuracy: predict(log_score, y)
    """
    def fit(self, X, y):
        self.mu = [np.mean(X[:, y == i], axis=1).reshape(-1, 1) for i in np.unique(y)]
        sigma = [np.dot(X[:, y == i] - self.mu[i], (X[:, y == i] - self.mu[i]).T) / X[:, y == i].shape[1] for i in np.unique(y)]
        self.sigma = [np.diag(np.diag(sigma[i])) for i in np.unique(y)]

class TiedCovarianceClassifier(MultivariateGaussianClassifier):
    """ Tied Covariance Classifier for multivariate data
        The classifier is based on the Gaussian pdf
        The classifier assumes that the data is normally distributed
        The classifier assumes that the covariance matrix is the same for all classes

        How to use:
        1. Create an instance of the class
        2. Call the fit method to train the model: fit(X, y)
        3. Call the log_gau_pdf method to compute the log of the Gaussian pdf: log_gau_pdf(X, mu[i], sigma)
        4. Call the predict method to compute the accuracy: predict(log_score, y)
    """
    def fit(self, X, y):
        self.mu = [np.mean(X[:, y == i], axis=1).reshape(-1, 1) for i in np.unique(y)]
        sigma = np.sum([np.dot(X[:, y == i] - self.mu[i], (X[:, y == i] - self.mu[i]).T) for i in np.unique(y)], axis=0) / X.shape[1]
        self.sigma = [sigma for _ in np.unique(y)]