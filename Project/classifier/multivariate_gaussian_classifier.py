import numpy as np
import scipy

class GaussianClassifier:
    def __init__(self):
        self.mean_per_class = None
        self.covariance_per_class = None
        self.diagonal_covariance = None
        self.cov_whithin = None
    
    @staticmethod
    def logarithmic_gau_pdf(sample, class_mean, class_covariance):
        d = sample.shape[1]  # Number of features
        sample = sample - class_mean.reshape(1, -1)
        inv_cov = np.linalg.inv(class_covariance)
        sign, log_det = np.linalg.slogdet(class_covariance)
        det_sign = sign * log_det
        quadratic_form = np.sum(sample @ inv_cov * sample, axis=1)
        log_pdf = -0.5 * d * np.log(2 * np.pi) - 0.5 * det_sign - 0.5 * quadratic_form
        return log_pdf

    def train_gaussian_classifier(self, data_train, label_train):
        self.mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
        self.covariance_per_class = [np.dot(data_train[:, label_train==i] - self.mean_per_class[i], (data_train[:, label_train==i] - self.mean_per_class[i]).T) / data_train[:, label_train==i].shape[1] for i in np.unique(label_train)]
        
    def train_naive_bayes_classifier(self, data_train, label_train):
        self.mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
        self.covariance_per_class = [np.dot(data_train[:, label_train==i] - self.mean_per_class[i], (data_train[:, label_train==i] - self.mean_per_class[i]).T) / data_train[:, label_train==i].shape[1] for i in np.unique(label_train)]
        self.diagonal_covariance = [np.diag(np.diag(self.covariance_per_class[i])) for i in np.unique(label_train)]
        
    def train_tied_covariance_classifier(self, data_train, label_train):
        self.mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
        self.cov_whithin = np.sum([np.dot(data_train[:, label_train==i] - self.mean_per_class[i], (data_train[:, label_train==i] - self.mean_per_class[i]).T) for i in np.unique(label_train)], axis=0) / data_train.shape[1]

    def predict_gaussian_classifier(self, log_score, label_test):
        log_score = log_score + np.log(1/3)  # Assuming 3 classes
        marginal_log_score = scipy.special.logsumexp(log_score, axis=0)
        posterior_log_score = log_score - marginal_log_score
        posterior_score = np.exp(posterior_log_score)
        acc = np.argmax(posterior_score, axis=0) == label_test
        accuracy = np.sum(acc)/label_test.shape[0]
        err_rate = 1 - accuracy
        return err_rate
