import numpy as np
import scipy.optimize as opt

class SVMClassifier:
    def __init__(self, C=10.0, K=1.0):
        self.C = C
        self.K = K
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        d, n = X.shape # d = number of features, n = number of samples
        zi = 2 * y - 1
        
        X_tilde = np.vstack([X, np.ones((1, n)) * self.K])
        hessian = np.dot(X_tilde.T, X_tilde) * np.outer(zi, zi)
        
        def objective(alpha):
            Ha = hessian @ alpha
            loss = 0.5 * alpha @ Ha - np.sum(alpha)
            return loss
        def gradient(alpha):
            return hessian @ alpha - np.ones(n)
        
        bounds = [(0, self.C) for _ in range(n)]
        alpha_opt, _, _ = opt.fmin_l_bfgs_b(objective, np.zeros(n), fprime=gradient, bounds=bounds, factr=1.0)
        w_hat = np.sum(alpha_opt * zi * X_tilde, axis=1)
        
        self.weight = w_hat[:-1]
        self.bias = w_hat[-1]* self.K
        dual_loss = -objective(alpha_opt)
        print(f"Optimal dual loss: {dual_loss}")
        
        def primal_loss(w_hat):
            reg = 0.5 * np.linalg.norm(w_hat) ** 2
            hinge_loss = np.maximum(0, 1 - zi * (w_hat.T @ X_tilde))
            return reg + self.C * np.sum(hinge_loss)
        
        primal_loss_value = primal_loss(w_hat)
        print(f"Primal loss: {primal_loss_value}")
        duality_gap = primal_loss_value - dual_loss
        print(f"Duality gap: {np.abs(duality_gap)}")

    def score(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        if self.weight is None:
            raise ValueError("Model not trained yet. Call fit() before score().")
        
        return self.weight.T @ X + self.bias
    
    def predict(self, X, y):
        z = self.score(X)
        y_pred = (z > 0) * 1
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy