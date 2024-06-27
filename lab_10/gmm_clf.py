import numpy as np
import scipy.special
import scipy.linalg

class GMM:
    def __init__(self, n_components=1, covariance_type='full', tol=1e-6):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.converged = False

    def logpdf_GAU_ND(self, x, mu, C):
        """ Compute the log of Gaussian PDF for N-dim data """
        size = len(x)
        det = np.linalg.det(C)
        norm_const = 1.0/ (np.power((2*np.pi), float(size)/2) * np.sqrt(det))
        x_mu = x - mu
        inv = np.linalg.inv(C)
        result = -0.5 * np.dot(x_mu.T, np.dot(inv, x_mu))
        return np.log(norm_const) + result

    def fit(self, X):
        """ Fit a GMM to the data using the EM algorithm """
        n_samples, n_features = X.shape
        # Initialization step
        self.means = np.random.rand(self.n_components, n_features)
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components

        log_likelihood = 0
        for iteration in range(100):
            # E-step
            responsibilities = np.zeros((n_samples, self.n_components))
            for i in range(self.n_components):
                pdf = np.array([self.logpdf_GAU_ND(x, self.means[i], self.covariances[i]) for x in X])
                responsibilities[:, i] = self.weights[i] * np.exp(pdf)
            sum_responsibilities = responsibilities.sum(axis=1)[:, np.newaxis]
            responsibilities /= sum_responsibilities

            # M-step
            weighted_data_sum = np.dot(responsibilities.T, X)
            for i in range(self.n_components):
                self.means[i] = weighted_data_sum[i] / responsibilities[:, i].sum()
                x_centered = X - self.means[i]
                self.covariances[i] = np.dot(responsibilities[:, i] * x_centered.T, x_centered) / responsibilities[:, i].sum()
                self.weights[i] = responsibilities[:, i].sum() / n_samples

            # Check convergence
            new_log_likelihood = np.sum(np.log(sum_responsibilities))
            if np.abs(new_log_likelihood - log_likelihood) <= self.tol:
                self.converged = True
                break
            log_likelihood = new_log_likelihood

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample."""
        log_prob = np.zeros(X.shape[0])
        for i in range(self.n_components):
            pdf = np.array([self.logpdf_GAU_ND(x, self.means[i], self.covariances[i]) for x in X])
            log_prob += self.weights[i] * np.exp(pdf)
        return np.log(log_prob)


