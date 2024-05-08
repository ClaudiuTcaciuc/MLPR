import numpy as np
import scipy

from bayesian_decision_evaluation import *

def main():
    # load the data
    llr = np.load("Data/commedia_llr_infpar.npy")
    y_true = np.load("Data/commedia_labels_infpar.npy")
    
    # impost the prior probability
    pi = 0.5
    print(f"Prior probability: {pi}")
    print(f"llr shape: {llr.shape}\n")
    
    # compute the cost matrix, prior class probability and threshold
    cost_matrix, prior_class_prob, threshold = binary_cost_matrix(pi)
    print(f"Cost matrix: \n{cost_matrix}")
    print(f"Prior class probability {prior_class_prob}")
    print(f"Threshold: {threshold}\n")
    
    y_pred_with_normal_threshold = np.where(llr > threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred_with_normal_threshold)
    
    acc_normal = accuracy(cm)
    print(f"Accuracy with normal threshold: {acc_normal}")
    
    # print the confusion matrix
    plot_confusion_matrix(cm)
    
    # compute the DCF
    DCF, P_fn, P_fp = compute_DCF(cm, cost_matrix, prior_class_prob)
    print(f"DCF: {DCF}")
    print(f"False negative probability: {P_fn}")
    print(f"False positive probability: {P_fp}\n")
    
    DCF_norm, P_fn, P_fp = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
    print(f"Normalized DCF: {DCF_norm}\n")
    
    # compute the minDCF and the best threshold
    minDCF, best_threshold = compute_minDCF(llr, y_true, pi, unique_labels=(0, 1))
    print(f"MinDCF: {minDCF}")
    print(f"Best threshold: {best_threshold}\n")
    
    y_pred = np.where(llr > best_threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred)
    
    acc = accuracy(cm)
    print(f"Accuracy with best threshold: {acc}")
    
    plot_confusion_matrix(cm)
    
    plot_ROC_curve(llr, y_true, cost_matrix, prior_class_prob, unique_labels=(0, 1))
    plot_bayes_error(llr, y_true, unique_labels=(0, 1))
    

if __name__ == "__main__":
    main()