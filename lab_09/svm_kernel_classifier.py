import numpy as np
import scipy.optimize as opt

class SVMClassifierKernel:
    def __init__(self, C=1.0, eps=1.0, kernel=None):
        self.kernel = kernel
        self.eps = eps
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
    
    def fit(self, X, y):
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        d, n = X.shape # d = number of features, n = number of samples
        
        zi = 2 * y - 1
        K = self.kernel(X, X) + self.eps
        H = np.outer(zi, zi) * K
        
        def objective(alpha):
            Ha = H @ alpha
            loss = 0.5 * alpha @ Ha - np.sum(alpha)
            return loss
        def gradient(alpha):
            return H @ alpha - np.ones(n)
        
        bounds = [(0, self.C) for _ in range(n)]
        alpha, _, _ = opt.fmin_l_bfgs_b(objective, np.zeros(n), fprime=gradient, bounds=bounds, factr=1.0)
        support_mask = alpha > 0
        
        self.alpha = alpha[support_mask]
        self.support_vectors = X[:, support_mask]
        self.support_vector_labels = zi[support_mask]
        print(f"SVM Kernel dual loss: {-objective(alpha):.4f}")
    
    def score(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        if self.alpha is None:
            raise ValueError("Model not trained yet. Call fit() before score().")
        
        n = X.shape[1]
        K = self.kernel(X, self.support_vectors) + self.eps
        z = np.sum(self.alpha * self.support_vector_labels * K, axis=1)
        return z

    def predict(self, X, y):
        z = self.score(X)
        y_pred = (z > 0) * 1
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy

class SVMClassifierPolyKernel(SVMClassifierKernel):
    def __init__(self, C=1.0, eps=1.0, degree=2, bias=1.0):
        kernel_func = self.kernel_func
        super().__init__(C, eps, kernel_func)
        self.degree = degree
        self.bias = bias
    
    def kernel_func(self, X1, X2):
        return (np.dot(X1.T, X2) + self.bias) ** self.degree

class SVMClassifierRBFKernel(SVMClassifierKernel):
    def __init__(self, C=1.0, eps=1.0, gamma=1.0):
        kernel_func = self.kernel_func
        super().__init__(C, eps, kernel_func)
        self.gamma = gamma
        
    def kernel_func(self, X1, X2):
        return np.exp(-self.gamma * np.sum((X1[:, :, np.newaxis] - X2[:, np.newaxis, :])**2, axis=0))