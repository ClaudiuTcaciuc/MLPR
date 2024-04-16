import numpy as np
import scipy
from sklearn import datasets

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

def logarithmic_gau_pdf(sample, class_mean, class_covariance):
    d = sample.shape[1]  # Number of features
    sample = sample - class_mean.reshape(1, -1)
    inv_cov = np.linalg.inv(class_covariance)
    sign, log_det = np.linalg.slogdet(class_covariance)
    det_sign = sign * log_det
    log_pdf = -0.5 * d * np.log(2 * np.pi) - 0.5 * det_sign - 0.5 * np.dot(np.dot(sample, inv_cov), sample.T).diagonal()
    return log_pdf

def train_gaussian_classifier(data_train, label_train):
    mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
    covariance_per_class = [np.dot(data_train[:, label_train==i] - mean_per_class[i], (data_train[:, label_train==i] - mean_per_class[i]).T) / data_train[:, label_train==i].shape[1] for i in np.unique(label_train)]
    
    return mean_per_class, covariance_per_class

def train_naive_bayes_classifier(data_train, label_train):
    mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
    covariance_per_class = [np.dot(data_train[:, label_train==i] - mean_per_class[i], (data_train[:, label_train==i] - mean_per_class[i]).T) / data_train[:, label_train==i].shape[1] for i in np.unique(label_train)]
    
    diagonal_covariance = [np.diag(np.diag(covariance_per_class[i])) for i in np.unique(label_train)]
    return mean_per_class, diagonal_covariance

def train_tied_covariance_classifier(data_train, label_train):
    mean_per_class = [np.mean(data_train[:, label_train==i], axis=1).reshape(-1, 1) for i in np.unique(label_train)]
    cov_whithin = np.sum([np.dot(data_train[:, label_train==i] - mean_per_class[i], (data_train[:, label_train==i] - mean_per_class[i]).T) for i in np.unique(label_train)], axis=0) / data_train.shape[1]
    return mean_per_class, cov_whithin

def multivariate_gaussian(data_train, label_train, data_test, label_test):
    log_score_mvg = np.zeros((len(np.unique(label_train)), data_test.shape[1]))
    log_score_nb = np.zeros((len(np.unique(label_train)), data_test.shape[1]))
    log_score_tied = np.zeros((len(np.unique(label_train)), data_test.shape[1]))
    
    mean_per_class_mvg, covariance_per_class_mvg = train_gaussian_classifier(data_train, label_train)
    mean_per_class_nb, variance_per_class_nb = train_naive_bayes_classifier(data_train, label_train)
    mean_per_class_tied, tied_covariance = train_tied_covariance_classifier(data_train, label_train)
       
    for i in np.unique(label_train):
        log_score_mvg[i] = logarithmic_gau_pdf(data_test.T, mean_per_class_mvg[i], covariance_per_class_mvg[i])
        log_score_nb[i] = logarithmic_gau_pdf(data_test.T, mean_per_class_nb[i], variance_per_class_nb[i])
        log_score_tied[i] = logarithmic_gau_pdf(data_test.T, mean_per_class_tied[i], tied_covariance)
    
    err_rate_mvg = predict_gaussian_classifier(log_score_mvg, label_test)
    err_rate_nb = predict_gaussian_classifier(log_score_nb, label_test)
    err_rate_tied = predict_gaussian_classifier(log_score_tied, label_test)
    print(f"Error rate MVG: {err_rate_mvg*100:.2f}%")
    print(f"Error rate NB: {err_rate_nb*100:.2f}%")
    print(f"Error rate Tied: {err_rate_tied*100:.2f}%")
    
def predict_gaussian_classifier(log_score, label_test):
    log_score = log_score + np.log(1/3)
    marginal_log_score = scipy.special.logsumexp(log_score, axis=0)
    posterior_log_score = log_score - marginal_log_score
    posterior_score = np.exp(posterior_log_score)
    acc = np.argmax(posterior_score, axis=0) == label_test
    accuracy = np.sum(acc)/label_test.shape[0]
    err_rate = 1 - accuracy
    return err_rate
    
def main():
    data, label = load_iris()
    data_train, label_train, data_test, label_test = split_data(data, label)
    multivariate_gaussian(data_train, label_train, data_test, label_test)
    
if __name__ == "__main__":
    main()
