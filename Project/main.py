import graph
import utils
import numpy as np

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
    pca_lda_computation(data, label, classes)

if __name__ == "__main__":
    main()