import graph
import utils
import numpy as np
from sklearn.linear_model import LogisticRegression

from classifier import multivariate_gaussian_classifier as mgc
from classifier import logistic_regression_classifier as lrc

def pca_lda_computation(data, label, classes):
    data_train, label_train, data_test, label_test = utils.split_data(data, label)
    
    pca_data_train, pca_selected_eigen_vectors = utils.pca(data_train, n_features=6, required_eigen_vectors=True)
    pca_data_test = np.dot(pca_selected_eigen_vectors.T, data_test)
    
    lda_data_train, lda_selected_eigen_vectors = utils.lda(pca_data_train, label_train, n_features=1, required_eigen_vectors=True)
    lda_data_test = np.dot(lda_selected_eigen_vectors.T, pca_data_test)
    
    graph.plot_histogram(data=lda_data_train, label=label_train, classes=classes)
    graph.plot_histogram(data=lda_data_test, label=label_test, classes=classes)
    
    threshold = (lda_data_train[0, label_train == 1].mean() + lda_data_train[0, label_train == 0].mean()) / 2
    #threshold = 10.0
    print(f'Threshold: {threshold}')
    predicted_values = np.zeros(shape=label_test.shape, dtype=np.int32)
    predicted_values[lda_data_test[0] < threshold] = 0
    predicted_values[lda_data_test[0] >= threshold] = 1
    
    count = np.sum(predicted_values == label_test)
    print(f'Accuracy: {count} out of {label_test.size} samples correctly classified. ({count/label_test.size*100:.2f}%)')
    
def main():
    data, label = utils.load_data()
    
    classes = {
        "Fake": "blue",
        "Real": "orange"
    }
    
    # graph.plot_histogram(data=data, label=label, classes=classes)
    # graph.plot_scatter(data=data, label=label, classes=classes)
    # graph.plot_correlation_matrix(data=data, label=label)
    # graph.plot_pca_explained_variance(data=data)
    # graph.plot_lda_histogram(data=data, label=label, classes=classes)
    # pca_lda_computation(data, label, classes)
    
    gaussian_classifier = mgc.GaussianClassifier()
    gaussian_classifier.train_gaussian_classifier(data, label)
    log_score = np.array([gaussian_classifier.logarithmic_gau_pdf(data.T, gaussian_classifier.mean_per_class[i], gaussian_classifier.covariance_per_class[i]) for i in np.unique(label)])
    err_rate_gaussian = gaussian_classifier.predict_gaussian_classifier(log_score, label)
    print(f"Error rate Gaussian: {err_rate_gaussian*100:.2f}%, Accuracy: {(1-err_rate_gaussian)*100:.2f}%")
    
    naive_bayes_classifier = mgc.GaussianClassifier()
    naive_bayes_classifier.train_naive_bayes_classifier(data, label)
    log_score = np.array([naive_bayes_classifier.logarithmic_gau_pdf(data.T, naive_bayes_classifier.mean_per_class[i], naive_bayes_classifier.diagonal_covariance[i]) for i in np.unique(label)])
    err_rate_naive_bayes = naive_bayes_classifier.predict_gaussian_classifier(log_score, label)
    print(f"Error rate Naive Bayes: {err_rate_naive_bayes*100:.2f}%, Accuracy: {(1-err_rate_naive_bayes)*100:.2f}%")
    
    tied_covariance_classifier = mgc.GaussianClassifier()
    tied_covariance_classifier.train_tied_covariance_classifier(data, label)
    log_score = np.array([tied_covariance_classifier.logarithmic_gau_pdf(data.T, tied_covariance_classifier.mean_per_class[i], tied_covariance_classifier.cov_whithin) for i in np.unique(label)])
    err_rate_tied_covariance = tied_covariance_classifier.predict_gaussian_classifier(log_score, label)
    print(f"Error rate Tied Covariance: {err_rate_tied_covariance*100:.2f}%, Accuracy: {(1-err_rate_tied_covariance)*100:.2f}%")
    
    label[label == 0] = -1
    logistic_regression = lrc.LogisticRegression(lambda_=0.1)
    logistic_regression.fit(data, label)
    err_rate_lr = logistic_regression.err_rate(data, label)
    print(f"Error rate LR: {err_rate_lr*100:.2f}%, Accuracy: {(1-err_rate_lr)*100:.2f}%")
    
    print("Using sklearn:")
    logistic_regression_sklearn = LogisticRegression(C=1/0.1, solver='lbfgs')
    logistic_regression_sklearn.fit(data.T, label)
    y_pred = logistic_regression_sklearn.predict(data.T)
    accuracy = np.mean(y_pred == label)
    print(f"Accuracy: {accuracy*100:.2f}%")

    
if __name__ == "__main__":
    main()