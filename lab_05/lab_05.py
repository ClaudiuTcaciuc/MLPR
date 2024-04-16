import numpy as np
import scipy
from sklearn import datasets

class GaussianClassifier:
    def __init__(self):
        pass
    
    @staticmethod
    def logarithmic_gau_pdf(sample, class_mean, class_covariance):
        d = sample.shape[1]  # Number of features
        sample = sample - class_mean.reshape(1, -1)
        inv_cov = np.linalg.inv(class_covariance)
        sign, log_det = np.linalg.slogdet(class_covariance)
        det_sign = sign * log_det
        # Use broadcasting to calculate the quadratic form efficiently
        quadratic_form = np.sum(sample @ inv_cov * sample, axis=1)
        log_pdf = -0.5 * d * np.log(2 * np.pi) - 0.5 * det_sign - 0.5 * quadratic_form
        return log_pdf

    @staticmethod
    def train_gaussian_classifier(data_train, label_train):
        mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
        covariance_per_class = [np.dot(data_train[:, label_train==i] - mean_per_class[i], (data_train[:, label_train==i] - mean_per_class[i]).T) / data_train[:, label_train==i].shape[1] for i in np.unique(label_train)]
        
        for i in np.unique(label_train):
            print(f"Class {i} mean vector:")
            print(mean_per_class[i])
            print(f"Class {i} covariance matrix:")
            print(covariance_per_class[i])
        
        return mean_per_class, covariance_per_class

    @staticmethod
    def train_naive_bayes_classifier(data_train, label_train):
        mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
        covariance_per_class = [np.dot(data_train[:, label_train==i] - mean_per_class[i], (data_train[:, label_train==i] - mean_per_class[i]).T) / data_train[:, label_train==i].shape[1] for i in np.unique(label_train)]
        diagonal_covariance = [np.diag(np.diag(covariance_per_class[i])) for i in np.unique(label_train)]
        
        for i in np.unique(label_train):
            print(f"Class {i} mean vector:")
            print(mean_per_class[i])
            print(f"Class {i} covariance matrix:")
            print(diagonal_covariance[i])
        
        return mean_per_class, diagonal_covariance

    @staticmethod
    def train_tied_covariance_classifier(data_train, label_train):
        mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
        cov_whithin = np.sum([np.dot(data_train[:, label_train==i] - mean_per_class[i], (data_train[:, label_train==i] - mean_per_class[i]).T) for i in np.unique(label_train)], axis=0) / data_train.shape[1]
        
        for i in np.unique(label_train):
            print(f"Class {i} mean vector:")
            print(mean_per_class[i])
            
        print(f"Class {i} covariance matrix:")
        print(cov_whithin)
        
        return mean_per_class, cov_whithin

    @staticmethod
    def predict_gaussian_classifier(log_score, label_test):
        log_score = log_score + np.log(1/3)
        marginal_log_score = scipy.special.logsumexp(log_score, axis=0)
        posterior_log_score = log_score - marginal_log_score
        posterior_score = np.exp(posterior_log_score)
        acc = np.argmax(posterior_score, axis=0) == label_test
        accuracy = np.sum(acc)/label_test.shape[0]
        err_rate = 1 - accuracy
        return err_rate

def load_iris():
    # Load iris dataset
    iris_data = datasets.load_iris()
    data, label = iris_data['data'].T, iris_data['target']
    return data, label

def split_data(data, label, perc=(2.0/3.0), seed=0):
    # Split the data 2/3 for train and 1/3 for test
    n_train = int(data.shape[1] * perc)
    np.random.seed(seed)
    index = np.random.permutation(data.shape[1])
    index_train = index[:n_train]
    index_test = index[n_train:]

    data_train = data[:, index_train]
    label_train = label[index_train]
    data_test = data[:, index_test]
    label_test = label[index_test]

    return data_train, label_train, data_test, label_test

def multivariate_gaussian(data_train, label_train, data_test, label_test):
    print("Training Gaussian classifier...")
    mean_per_class_mvg, covariance_per_class_mvg = GaussianClassifier.train_gaussian_classifier(data_train, label_train)
    print("Training Naive Bayes classifier...")
    mean_per_class_nb, variance_per_class_nb = GaussianClassifier.train_naive_bayes_classifier(data_train, label_train)
    print("Training tied covariance classifier...")
    mean_per_class_tied, tied_covariance = GaussianClassifier.train_tied_covariance_classifier(data_train, label_train)
       
    print("Testing classifiers...")
    log_score_mvg = np.zeros((len(np.unique(label_train)), data_test.shape[1]))
    log_score_nb = np.zeros((len(np.unique(label_train)), data_test.shape[1]))
    log_score_tied = np.zeros((len(np.unique(label_train)), data_test.shape[1]))
    
    for i in np.unique(label_train):
        log_score_mvg[i] = GaussianClassifier.logarithmic_gau_pdf(data_test.T, mean_per_class_mvg[i], covariance_per_class_mvg[i])
        log_score_nb[i] = GaussianClassifier.logarithmic_gau_pdf(data_test.T, mean_per_class_nb[i], variance_per_class_nb[i])
        log_score_tied[i] = GaussianClassifier.logarithmic_gau_pdf(data_test.T, mean_per_class_tied[i], tied_covariance)
    
    err_rate_mvg = GaussianClassifier.predict_gaussian_classifier(log_score_mvg, label_test)
    err_rate_nb = GaussianClassifier.predict_gaussian_classifier(log_score_nb, label_test)
    err_rate_tied = GaussianClassifier.predict_gaussian_classifier(log_score_tied, label_test)
    
    print(f"Error rate MVG: {err_rate_mvg*100:.2f}%")
    print(f"Error rate NB: {err_rate_nb*100:.2f}%")
    print(f"Error rate Tied: {err_rate_tied*100:.2f}%")
    
def main():
    print("Loading iris dataset...")
    data, label = load_iris()
    print("Splitting data...")
    data_train, label_train, data_test, label_test = split_data(data, label)
    multivariate_gaussian(data_train, label_train, data_test, label_test)
    
if __name__ == "__main__":
    main()
