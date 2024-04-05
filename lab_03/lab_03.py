import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import datasets

def load_iris():
    # Load iris dataset
    iris_data = datasets.load_iris()
    data, label = iris_data['data'].T, iris_data['target']
    return data, label

def load_iris_binary():
    # Load binary iris dataset
    # setosa = 0, versicolor = 1, virginica = 2
    iris_data = datasets.load_iris()
    data, label = iris_data['data'].T, iris_data['target']
    
    data = data[:, label != 0]
    label = label[label != 0]
    label[label == 1] = 0
    label[label == 2] = 1
    return data, label

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

def pca(data, n_features=4):
    # Compute the PCA decomposition on n features
    
    mean = np.mean(data, axis=1).reshape(-1, 1)
    centered_data = data - mean
    cov = np.dot(centered_data, centered_data.T) / data.shape[1]
    
    eigen_values, eigen_vectors = scipy.linalg.eigh(cov)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vectors[:, sorted_index]
    selected_eigen_vectors = sorted_eigen_vectors[:, :n_features]
    
    new_data = np.dot(selected_eigen_vectors.T, data)
    return new_data

def compute_Sw_Sb(data, label):
    data_class = [data[:, label==i] for i in np.unique(label)]
    sample_class = [data_class[i].shape[1] for i in np.unique(label)]
    
    mean = np.mean(data, axis=1).reshape(-1, 1)
    mean_class = [np.mean(data_class[i], axis=1).reshape(-1, 1) for i in np.unique(label)]
    S_w, S_b = 0, 0
    for i in np.unique(label):
        data_c = data_class[i] - mean_class[i]
        cov_c = np.dot(data_c, data_c.T) / data_c.shape[1]
        S_w += sample_class[i] * cov_c
        diff = mean_class[i] - mean
        S_b += sample_class[i] * np.dot(diff, diff.T)
    S_w /= data.shape[1]
    S_b /= data.shape[1]
    return S_w, S_b

def lda(data, label, n_features=3):
    # Compute the LDA decomposition on n features
    Sw, Sb = compute_Sw_Sb(data=data, label=label)
    
    _, eigen_vectors = scipy.linalg.eigh(Sb, Sw)
    selected_eigen_vectors = eigen_vectors[:, ::-1][:, :n_features]
    lda_data = np.dot(selected_eigen_vectors.T, data)
    
    return lda_data, selected_eigen_vectors

def plot_scatter(data, label, classes, features=(0, 1)):
    plt.figure()
    
    n_features = np.unique(features).size
    
    for i in range(n_features-1):
        for j in range(i+1, n_features):
            for k in range(np.unique(label).size):
                plt.scatter(data[features[i], label == k], data[features[i], label == k], label=classes[k])
            plt.xlabel(f'Feature {features[i]+1}')
            plt.ylabel(f'Feature {features[j]+1}')
            plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def plot_histogram(data, label, classes, features=(0, 1)):
    plt.figure()
    
    n_features = np.unique(features).size
    
    for i in range(n_features):
        plt.subplot(1, n_features, i+1)
        for j in range(np.unique(label).size):
            plt.hist(data[features[i], label == j], alpha=0.5, label=classes[j], density=True, ec='black', bins=5)
        plt.xlabel(f'Feature {features[i]+1}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def main():
    data, label = load_iris_binary()
    data_train, label_train, data_test, label_test = split_data(data, label)
    
    lda_data_train, lda_selected_eigenvalues = lda(data_train, label_train, n_features=1)
    lda_data_test = np.dot(lda_selected_eigenvalues.T, data_test)
    
    plot_histogram(lda_data_train, label_train, ['versicolor', 'virginica'], features=[0])
    plot_histogram(lda_data_test, label_test, ['versicolor', 'virginica'], features=[0])
    
    threshold = (lda_data_train[0, label_train == 1].mean() + lda_data_train[0, label_train == 0].mean()) / 2
    predicted_values = np.zeros(shape=label_test.shape, dtype=np.int32)
    predicted_values[lda_data_test[0] < threshold] = 0
    predicted_values[lda_data_test[0] >= threshold] = 1
    
    count = np.sum(predicted_values == label_test)
    print(f'Accuracy: {count} out of {label_test.size} samples correctly classified.')
    
if __name__ == "__main__":
    main()
