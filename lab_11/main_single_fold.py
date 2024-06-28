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
    
    prior = 0.2
    
    print()
    # Calibration single-fold approach
    print("Calibration single-fold approach")
    scal1, sval1 = score_s1[::3], np.hstack([score_s1[1::3], score_s1[2::3]])
    scal2, sval2 = score_s2[::3], np.hstack([score_s2[1::3], score_s2[2::3]])
    labels_cal, labels_val = labels[::3], np.hstack([labels[1::3], labels[2::3]])
    
    clf_s1 = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
    clf_s1.fit(scal1.reshape(1, -1), labels_cal)
    
    clf_s2 = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
    clf_s2.fit(scal2.reshape(1, -1), labels_cal)
    
    print("\tSystem 1 not calibrated")
    compute_statistics(sval1, labels_val, prior, unique_labels=(0, 1))
    print("\tSystem 1 calibrated")
    score_s1_cal = clf_s1.score(sval1.reshape(1, -1)) - np.log(prior / (1 - prior))
    compute_statistics(score_s1_cal, labels_val, prior, unique_labels=(0, 1))
    print("\tSystem 1 evaluation not calibrated")
    compute_statistics(eval_scores_s1, eval_labels, prior, unique_labels=(0, 1))
    print("\tSystem 1 evaluation calibrated")
    score_s1_eval_cal = clf_s1.score(eval_scores_s1.reshape(1, -1)) - np.log(prior / (1 - prior))
    compute_statistics(score_s1_eval_cal, eval_labels, prior, unique_labels=(0, 1))
    
    print()
    
    print("\tSystem 2 not calibrated")
    compute_statistics(sval2, labels_val, prior, unique_labels=(0, 1))
    print("\tSystem 2 calibrated")
    score_s2_cal = clf_s2.score(sval2.reshape(1, -1)) - np.log(prior / (1 - prior))
    compute_statistics(score_s2_cal, labels_val, prior, unique_labels=(0, 1))
    print("\tSystem 2 evaluation not calibrated")
    compute_statistics(eval_scores_s2, eval_labels, prior, unique_labels=(0, 1))
    print("\tSystem 2 evaluation calibrated")
    score_s2_eval_cal = clf_s2.score(eval_scores_s2.reshape(1, -1)) - np.log(prior / (1 - prior))
    compute_statistics(score_s2_eval_cal, eval_labels, prior, unique_labels=(0, 1))
    
    print()
    print("Fusion single-fold approach")
    
    fused_sys = np.vstack([scal1, scal2])
    fused_sys_val = np.vstack([sval1, sval2])
    clf_fusion = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
    clf_fusion.fit(fused_sys, labels_cal)
    score_fusion = clf_fusion.score(fused_sys_val) - np.log(prior / (1 - prior))
    print("\tFusion validation")
    compute_statistics(score_fusion, labels_val, prior, unique_labels=(0, 1))
    
    fused_sys_eval = np.vstack([eval_scores_s1, eval_scores_s2])
    score_fusion_eval = clf_fusion.score(fused_sys_eval) - np.log(prior / (1 - prior))
    print("\tFusion evaluation")
    compute_statistics(score_fusion_eval, eval_labels, prior, unique_labels=(0, 1))
    
if __name__ == "__main__":
    main()
