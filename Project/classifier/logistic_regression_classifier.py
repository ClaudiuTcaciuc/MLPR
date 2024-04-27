import numpy as np
import scipy.optimize as opt

class LogisticRegression:
    def __init__(self, lambda_=0):
        self.weights = None
        self.lambda_ = lambda_
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def logreg_obj(self, v, X, y):
        _, n = X.shape
        w, b = v[0:-1], v[-1]
        z = w.T @ X + b
        loss = np.logaddexp(0, -y * z)
        reg = (self.lambda_ / 2) * np.sum(w ** 2)
        J = (1 / n) * np.sum(loss) + reg
        return J
    
    def fit(self, X, y):
        d, _ = X.shape
        x0_train = np.zeros(d + 1)
        
        x_opt, _, _ = opt.fmin_l_bfgs_b(self.logreg_obj, x0_train, args=(X, y), approx_grad=True)
        self.weights = x_opt
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit() before predict().")
        z = self.weights[:-1].T @ X + self.weights[-1] # score
        y_pred = np.sign(self.sigmoid(z) - 0.5)
        return y_pred
    
    def err_rate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return 1 - accuracy
