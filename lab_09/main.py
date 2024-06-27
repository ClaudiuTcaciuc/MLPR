import numpy as np
from bayesian_decision_evaluation import *
from sklearn.datasets import load_iris
from svm_classifier import SVMClassifier
from svm_kernel_classifier import SVMClassifierPolyKernel, SVMClassifierRBFKernel

def load_iris_binary():
    D, L = load_iris()['data'].T, load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L

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
    D, L = load_iris_binary()
    data_train, label_train, data_test, label_test = split_data(D, L)
    # print(data_train.shape, label_train.shape) # (4, 66) (66,)
    C = [0.1, 1, 10]
    K = [1, 10]
    for k in K:
        for c in C:
            print(f"\tC={c}, K={k}")
            model = SVMClassifier(C=c, K=k)
            model.fit(data_train, label_train)
            accuracy = model.predict(data_test, label_test)
            print(f"Accuracy: {(accuracy)*100:.1f}, Error rate: {(1 - accuracy)*100:.1f}")
            score = model.score(data_test)
            compute_statistics(score, label_test, 0.5, unique_labels=(0, 1))
    
    C = 1.0
    eps = [0.0, 1.0]
    degree = 2
    bias = [0.0, 1.0]
    # Polynomial Kernel
    for e in eps:
        for b in bias:
            print(f"\tEps={e}, Bias={b}")
            model = SVMClassifierPolyKernel(C=C, eps=e, degree=degree, bias=b)
            model.fit(data_train, label_train)
            acc = model.predict(data_test, label_test)
            print(f"Accuracy: {(acc)*100:.1f}, Error rate: {(1 - acc)*100:.1f}")
            score = model.score(data_test)
            compute_statistics(score, label_test, 0.5, unique_labels=(0, 1))

    # RBF Kernel
    lamdas = [1.0, 10.0]
    for e in eps:
        for l in lamdas:
            print(f"\tEps={e}, Lambda={l}")
            model = SVMClassifierRBFKernel(C=C, eps=e, gamma=l)
            model.fit(data_train, label_train)
            acc = model.predict(data_test, label_test)
            print(f"Accuracy: {(acc)*100:.1f}, Error rate: {(1 - acc)*100:.1f}")
            score = model.score(data_test)
            compute_statistics(score, label_test, 0.5, unique_labels=(0, 1))
    

if __name__ == "__main__":
    main()