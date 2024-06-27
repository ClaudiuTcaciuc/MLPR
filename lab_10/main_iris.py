import numpy as np
from gmm_clf import GMM
from bayesian_decision_evaluation import *
import json
import time
from sklearn.datasets import load_iris

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

def compute_statistics(llr, y_true, prior, unique_labels=None):
    cost_matrix, prior_class_prob, threshold = binary_cost_matrix(prior)
    
    min_DCF, _ = compute_minDCF(llr, y_true, prior, unique_labels)
    y_pred = np.where(llr >= threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred, unique_labels)
    DCF, _, _ = compute_DCF(cm, cost_matrix, prior_class_prob)
    DCF_norm, _, _ = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
    
    print(f"MinDCF: {min_DCF:.4f}, DCF: {DCF:.4f}, Normalized DCF: {DCF_norm:.4f}\n")

def main():
    # IRIS
    D, L = load_iris()['data'].T, load_iris()['target']
    DTR, LTR, DTE, LTE = split_data(D, L)
    
    for covariance_type in ['full', 'diagonal', 'tied']:
        for n_components in [1, 2, 4, 8, 16]:
            clf = GMM(n_components=n_components, covariance_type=covariance_type, verbose=False, psiEig=0.01)
            clf.fit(DTR, LTR, n_features=3)
            score = clf.score(DTE)
            preds = clf.predict(DTE)
            error_rate = (preds != LTE).sum() / len(LTE)
            
            print(f"IRIS - {clf.covariance_type} - {clf.n_components}, Error rate: {(error_rate)*100:.2f}%")
            
    print()
    print('Binary task')
    D, L = np.load('Data/ext_data_binary.npy'), np.load('Data/ext_data_binary_labels.npy')
    DTR, LTR, DTE, LTE = split_data(D, L)
    
    for covariance_type in ['full', 'diagonal', 'tied']:
        for n_components in [1, 2, 4, 8, 16]:
            print(f"Binary task - {covariance_type} - {n_components}")
            clf = GMM(n_components=n_components, covariance_type=covariance_type, verbose=False, psiEig=0.01)
            clf.fit(DTR, LTR, n_features=2)
            
            score = clf.score_binary(DTE)
            compute_statistics(score, LTE, 0.5)  
    
if __name__ == '__main__':
    main()

