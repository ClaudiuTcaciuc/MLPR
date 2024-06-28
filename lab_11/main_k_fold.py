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
    
    k_fold = 5
    prior = 0.2
    
    def extract_fold(X, idx):
        return np.hstack([X[jdx::k_fold] for jdx in range(k_fold) if jdx != idx]), X[idx::k_fold]
    
    def calibrate_system(score_s1, labels, prior):
        calibrated_scores = []
        calibrated_labels = []
        for i in range(k_fold):
            score_cal, score_val = extract_fold(score_s1, i)
            labels_cal, labels_val = extract_fold(labels, i)
            
            clf = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
            # print(f"\t\tCalibration fold {i + 1}\n")
            clf.fit(score_cal.reshape(1, -1), labels_cal)
            calibrated_sval = clf.score(score_val.reshape(1, -1)) - np.log(prior / (1 - prior))
            calibrated_scores.append(calibrated_sval)
            calibrated_labels.append(labels_val)
        
        calibrated_scores = np.hstack(calibrated_scores)
        calibrated_labels = np.hstack(calibrated_labels)

        return calibrated_scores, calibrated_labels

    calibrated_scores_s1, labels_calibrated_s1 = calibrate_system(score_s1, labels, prior)
    calibrated_scores_s2, labels_calibrated_s2 = calibrate_system(score_s2, labels, prior)
    
    clf_cal_s1 = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels==1), n_F=np.sum(labels==0))
    clf_cal_s1.fit(score_s1.reshape(1, -1), labels)
    
    clf_cal_s2 = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels==1), n_F=np.sum(labels==0))
    clf_cal_s2.fit(score_s2.reshape(1, -1), labels)
    
    eval_scores_s1_cal = clf_cal_s1.score(eval_scores_s1.reshape(1, -1)) - np.log(prior / (1 - prior))
    eval_scores_s2_cal = clf_cal_s2.score(eval_scores_s2.reshape(1, -1)) - np.log(prior / (1 - prior))
    
    # fig, axs = plt.subplots(2, 2, figsize=(15, 5))  # Creates a figure with 2 subplots
    print(f"\tSystem 1 with prior: {prior}")
    compute_statistics(score_s1, labels, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=score_s1, y_true=labels, unique_labels=(0, 1), ax=axs[0, 0])
    # axs[0, 0].set_title('System 1 Bayes error')
    print(f"\tSystem 1 after calibration with prior: {prior}")
    compute_statistics(calibrated_scores_s1, labels_calibrated_s1, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=calibrated_scores_s1, y_true=labels_calibrated_s1, unique_labels=(0, 1), ax=axs[0, 1])
    # axs[0, 1].set_title('System 1 Calibration Bayes error')
    print("\tSystem 1 evaluation with prior: 0.2")
    compute_statistics(eval_scores_s1, eval_labels, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=eval_scores_s1, y_true=eval_labels, unique_labels=(0, 1), ax=axs[1, 0])
    # axs[1, 0].set_title('System 1 Evaluation Bayes error')
    print(f"\tSystem 1 evaluation after calibration with prior: {prior}")
    compute_statistics(eval_scores_s1_cal, eval_labels, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=eval_scores_s1_cal, y_true=eval_labels, unique_labels=(0, 1), ax=axs[1, 1])
    # axs[1, 1].set_title('System 1 Evaluation Calibration Bayes error')
    # plt.show()
    
    print()
    # fig, axs = plt.subplots(2, 2, figsize=(15, 5))  # Creates a figure with 2 subplots
    print(f"\tSystem 2 with prior: {prior}")
    compute_statistics(score_s2, labels, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=score_s2, y_true=labels, unique_labels=(0, 1), ax=axs[0, 0])
    # axs[0, 0].set_title('System 2 Bayes error')
    print(f"\tSystem 2 after calibration with prior: {prior}")
    compute_statistics(calibrated_scores_s2, labels_calibrated_s2, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=calibrated_scores_s2, y_true=labels_calibrated_s2, unique_labels=(0, 1), ax=axs[0, 1])
    # axs[0, 1].set_title('System 2 Calibration Bayes error')
    print("\tSystem 2 evaluation with prior: 0.2")
    compute_statistics(eval_scores_s2, eval_labels, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=eval_scores_s2, y_true=eval_labels, unique_labels=(0, 1), ax=axs[1, 0])
    # axs[1, 0].set_title('System 2 Evaluation Bayes error')
    print(f"\tSystem 2 evaluation after calibration with prior: {prior}")
    compute_statistics(eval_scores_s2_cal, eval_labels, prior, unique_labels=(0, 1))
    # plot_bayes_error(scores=eval_scores_s2_cal, y_true=eval_labels, unique_labels=(0, 1), ax=axs[1, 1])
    # axs[1, 1].set_title('System 2 Evaluation Calibration Bayes error')
    # plt.show()
    
    print()
    print("Fusion")
    
    fused_scores = []
    fused_labels = []
    
    for idx in range(k_fold):
        score_cal_s1, score_val_s1 = extract_fold(score_s1, idx)
        score_cal_s2, score_val_s2 = extract_fold(score_s2, idx)
        labels_cal, labels_val = extract_fold(labels, idx)
        
        score_cal = np.vstack([score_cal_s1, score_cal_s2])
        score_val = np.vstack([score_val_s1, score_val_s2])
        
        clf = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
        clf.fit(score_cal, labels_cal)
        fused_score = clf.score(score_val) - np.log(prior / (1 - prior))
        fused_scores.append(fused_score)
        fused_labels.append(labels_val)
    
    fused_scores = np.hstack(fused_scores)
    fused_labels = np.hstack(fused_labels)
    
    print(f"\tFusion with prior: {prior}")
    compute_statistics(fused_scores, fused_labels, prior, unique_labels=(0, 1))
    
    score_matrix = np.vstack([score_s1, score_s2])
    clf_fusion = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels==1), n_F=np.sum(labels==0))
    clf_fusion.fit(score_matrix, labels)
    
    score_eval_matrix = np.vstack([eval_scores_s1, eval_scores_s2])
    fused_scores_eval = clf_fusion.score(score_eval_matrix) - np.log(prior / (1 - prior))
    
    print(f"\tFusion evaluation with prior: {prior}")
    compute_statistics(fused_scores_eval, eval_labels, prior, unique_labels=(0, 1))
    
if __name__ == "__main__":
    main()
