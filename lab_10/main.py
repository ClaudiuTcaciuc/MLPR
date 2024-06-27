import numpy as np
from gmm_clf import GMM
from bayesian_decision_evaluation import *
import json
import time

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

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def main():
    t1 = time.time()
    clf = GMM(verbose=True, tol=1e-6)
    
    X = np.load('Data/GMM_data_4D.npy')
    gmm = load_gmm('Data/GMM_4D_3G_init.json')
    llPrecomputed = np.load('Data/GMM_4D_3G_init_ll.npy')
    ll = clf.logpdf_GMM(X, gmm)
    print (np.abs(ll-llPrecomputed).max()) # Check max absolute difference
    
    X = np.load('Data/GMM_data_1D.npy')
    gmm = load_gmm('Data/GMM_1D_3G_init.json')
    llPrecomputed = np.load('Data/GMM_1D_3G_init_ll.npy')
    ll = clf.logpdf_GMM(X, gmm)
    print (np.abs(ll-llPrecomputed).max()) # Check max absolute difference
    
    print()
    print('***** EM - 4D *****')
    print()
    X = np.load('Data/GMM_data_4D.npy')
    gmm = load_gmm('Data/GMM_4D_3G_init.json')
    gmm = clf.train_GMM_EM(X, gmm)
    print ('Final average ll: %.8e' % clf.logpdf_GMM(X, gmm).mean())
    
    print()
    print('***** EM - 1D *****')
    print()
    X = np.load('Data/GMM_data_1D.npy')
    gmm = load_gmm('Data/GMM_1D_3G_init.json')
    gmm = clf.train_GMM_EM(X, gmm)
    print ('Final average ll: %.8e' % clf.logpdf_GMM(X, gmm).mean())
    
    plt.figure()
    plt.hist(X.ravel(), 25, density=True) # Pay attention to the shape of X: X is a data matrix, so it's a 1 x N array, not a 1-D array
    _X = np.linspace(-10, 5, 1000) # Plot gmm density in range (-10, 5) - x-data for the plot
    plt.plot(_X.ravel(), np.exp(clf.logpdf_GMM(_X.reshape(1, -1), gmm))) # Pay attention to the shape of _X: for plotting _X should be a 1-D array, for logpdf_GMM it should be a 1 x N matrix with one-dimensional samples
    # plt.show()
    
    print()
    print('***** LBG EM - 4D *****')
    print()
    X = np.load('Data/GMM_data_4D.npy')
    gmm = clf.train_GMM_LBG_EM(X, 4)
    print ('LBG + EM - final average ll: %.8e (%d components)' % (clf.logpdf_GMM(X, gmm).mean(), len(gmm)))
    print ('LBG + EM - final average ll - pretrained model: %.8e' % (clf.logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')).mean()))
    #print(gmm) # you can print the gmms
    #print(load_gmm('Data/GMM_4D_4G_EM_LBG.json')) # you can print the gmms
    print ('Max absolute ll difference w.r.t. pre-trained model over all training samples:', (np.abs(clf.logpdf_GMM(X, gmm) - clf.logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')))).max())
    
    print()
    print('***** LBG EM - 1D *****')
    print()
    X = np.load('Data/GMM_data_1D.npy')
    gmm = clf.train_GMM_LBG_EM(X, 4)
    print ('LBG + EM - final average ll: %.8e (%d components)' % (clf.logpdf_GMM(X, gmm).mean(), len(gmm)))
    print ('LBG + EM - final average ll - pretrained model: %.8e' % (clf.logpdf_GMM(X, load_gmm('Data/GMM_1D_4G_EM_LBG.json')).mean()))
    #print(gmm) # you can print the gmms
    #print(load_gmm('Data/GMM_1D_4G_EM_LBG.json')) # you can print the gmms
    print ('Max absolute ll difference w.r.t. pre-trained model over all training samples:', (np.abs(clf.logpdf_GMM(X, gmm) - clf.logpdf_GMM(X, load_gmm('Data/GMM_1D_4G_EM_LBG.json')))).max())

    plt.figure()
    plt.hist(X.ravel(), 25, density=True) # Pay attention to the shape of X: X is a data matrix, so it's a 1 x N array, not a 1-D array
    _X = np.linspace(-10, 5, 1000) # Plot gmm density in range (-10, 5) - x-data for the plot
    plt.plot(_X.ravel(), np.exp(clf.logpdf_GMM(_X.reshape(1, -1), gmm)), 'r') # Pay attention to the shape of _X: for plotting _X should be a 1-D array, for logpdf_GMM it should be a 1 x N matrix with one-dimensional samples
    # plt.show()
    
    print()
    print('***** LBG EM - 4D - Eigenvalue Theshold *****')
    print()
    X = np.load('Data/GMM_data_4D.npy')
    gmm = clf.train_GMM_LBG_EM(X, 4)
    print ('LBG + EM - final average ll: %.8e (%d components)' % (clf.logpdf_GMM(X, gmm).mean(), len(gmm)))
    print ('LBG + EM - final average ll - pretrained model: %.8e' % (clf.logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')).mean()))
    #print(gmm) # you can print the gmms
    #print(load_gmm('Data/GMM_4D_4G_EM_LBG.json')) # you can print the gmms
    print ('Max absolute ll difference w.r.t. pre-trained model over all training samples:', (np.abs(clf.logpdf_GMM(X, gmm) - clf.logpdf_GMM(X, load_gmm('Data/GMM_4D_4G_EM_LBG.json')))).max())

    
    t2 = time.time()
    print(f"Time: {t2 - t1:.2f} seconds")
    
if __name__ == "__main__":
    main()