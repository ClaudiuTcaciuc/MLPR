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
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 5))  # Creates a figure with 2 subplots
    
    print()
    # Calibration single-fold approach
    print("Calibration single-fold approach")
    scal1, sval1 = score_s1[::3], np.hstack([score_s1[1::3], score_s1[2::3]])
    scal2, sval2 = score_s2[::3], np.hstack([score_s2[1::3], score_s2[2::3]])
    labels_cal, labels_val = labels[::3], np.hstack([labels[1::3], labels[2::3]])
    
    prior = 0.2
    print(f"\tSystem 1 with prior: {prior}")
    compute_statistics(sval1, labels_val, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=sval1, y_true=labels_val, unique_labels=(0, 1), ax=axs[0, 0])
    axs[0, 0].set_title('System 1 Bayes error')
    
    print(f"\tSystem 2 with prior: {prior}")
    compute_statistics(sval2, labels_val, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=sval2, y_true=labels_val, unique_labels=(0, 1), ax=axs[0, 2])
    axs[0, 2].set_title('System 2 Bayes error')
    
    print()
    print(f"\t System 1 evaluation with prior: {prior}")
    compute_statistics(eval_scores_s1, eval_labels, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=eval_scores_s1, y_true=eval_labels, unique_labels=(0, 1), ax=axs[0, 1])
    axs[0, 1].set_title('System 1 Evaluation Bayes error')
    
    print(f"\t System 2 evaluation with prior: {prior}")
    compute_statistics(eval_scores_s2, eval_labels, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=eval_scores_s2, y_true=eval_labels, unique_labels=(0, 1), ax=axs[0, 3])
    axs[0, 3].set_title('System 2 Evaluation Bayes error')
    
    print() 
    clf = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
    clf.fit(scal1.reshape(1, -1), labels_cal)
    score = clf.score(sval1.reshape(1, -1)) - np.log(prior / (1 - prior))
    print(f"\tSystem 1 with prior: {prior} after calibration")
    compute_statistics(score, labels_val, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=score, y_true=labels_val, unique_labels=(0, 1), ax=axs[1, 0])
    axs[1, 0].set_title('System 1 Calibration Bayes error')
    
    print()
    score = clf.score(eval_scores_s1.reshape(1, -1)) - np.log(prior / (1 - prior))
    print(f"\tSystem 1 evaluation with prior: {prior} after calibration")
    compute_statistics(score, eval_labels, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=score, y_true=eval_labels, unique_labels=(0, 1), ax=axs[1, 1])
    axs[1, 1].set_title('System 1 Evaluation Calibration Bayes error')
    
    clf.fit(scal2.reshape(1, -1), labels_cal)
    score = clf.score(sval2.reshape(1, -1)) - np.log(prior / (1 - prior))
    print(f"\tSystem 2 with prior: {prior} after calibration")
    compute_statistics(score, labels_val, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=score, y_true=labels_val, unique_labels=(0, 1), ax=axs[1, 2])
    axs[1, 2].set_title('System 2 Calibration Bayes error')
    
    score = clf.score(eval_scores_s2.reshape(1, -1)) - np.log(prior / (1 - prior))
    print(f"\tSystem 2 evaluation with prior: {prior} after calibration")
    compute_statistics(score, eval_labels, prior, unique_labels=(0, 1))
    plot_bayes_error(scores=score, y_true=eval_labels, unique_labels=(0, 1), ax=axs[1, 3])
    axs[1, 3].set_title('System 2 Evaluation Calibration Bayes error')
    
    plt.show()
    
if __name__ == "__main__":
    main()
