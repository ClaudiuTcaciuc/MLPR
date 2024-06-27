import numpy as np
from bayesian_decision_evaluation import *
from logistic_regression_classifier import LogisticRegressionWeighted

def compute_statistics(llr, y_true, prior, unique_labels=None):
    cost_matrix, prior_class_prob, threshold = binary_cost_matrix(prior)
    
    min_DCF, _ = compute_minDCF(llr, y_true, prior, unique_labels)
    y_pred = np.where(llr >= threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred, unique_labels)
    DCF, _, _ = compute_DCF(cm, cost_matrix, prior_class_prob)
    DCF_norm, _, _ = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
    
    print(f"MinDCF: {min_DCF:.3f}, DCF: {DCF:.3f}, Normalized DCF: {DCF_norm:.3f}\n")

def plot_bayes_error(scores, y_true, unique_labels, ax):
    eff_prior_log_odds = np.linspace(-3, 3, 21)
    eff_prior = 1 / (1 + np.exp(-eff_prior_log_odds))
    normalized_DCF = []
    normalized_minDCF = []
    
    for pi in eff_prior:
        cost_matrix, prior_class_prob, threshold = binary_cost_matrix(pi)
        minDCF, _ = compute_minDCF(scores, y_true, pi, unique_labels)
        normalized_minDCF.append(minDCF)
        y_pred = np.where(scores > threshold, unique_labels[1], unique_labels[0])
        cm = confusion_matrix(y_true, y_pred, unique_labels)
        DCF, _, _ = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
        normalized_DCF.append(DCF)
    
    ax.plot(eff_prior_log_odds, normalized_DCF, label='Normalized DCF', color='red')
    ax.plot(eff_prior_log_odds, normalized_minDCF, label='Normalized minDCF', color='blue')
    ax.set_xlabel('Effective prior probability')
    ax.set_ylabel('Detection Cost Function')
    ax.set_ylim([0, 1.1])
    ax.set_xlim([-3, 3])
    ax.legend()

def main():
    score_s1 = np.load('Data/scores_1.npy')
    score_s2 = np.load('Data/scores_2.npy')
    
    eval_scores_s1 = np.load('Data/eval_scores_1.npy')
    eval_scores_s2 = np.load('Data/eval_scores_2.npy')
    
    labels = np.load('Data/labels.npy')
    eval_labels = np.load('Data/eval_labels.npy')
    
if __name__ == "__main__":
    main()
