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
    iris_data = datasets.load_iris()
    data, label = iris_data['data'].T, iris_data['target']
    
    data = data[:, label != 0]
    label = label[label != 0]
    label[label == 2] = 0
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
    
    cov = np.cov(data)
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
    
    return lda_data

def plot_iris_scatter(data, label, first_f, second_f):
    # Plot scatter plots for iris dataset features
    plt.figure(figsize=(10, 10))
    classes = ['versicolor', 'virginica']
    colors = ['orange', 'green']
    
    for k in range(3):
        plt.scatter(data[first_f, label == k], data[second_f, label == k], label=classes[k], s=20, c=colors[k])
    plt.xlabel(f'Feature {second_f+1}')
    plt.ylabel(f'Feature {first_f+1}')
    plt.legend(loc='best')
    plt.show()

def plot_iris_histogram(data, label):
    # Plot histograms for iris dataset features
    plt.figure()
    classes = ['virginica', 'versicolor']
    colors = ['blue', 'orange']
    
    for i in range(1):
        print(data.shape)
        plt.hist(data[i, label == 0], alpha=0.5, label=classes[0], density=True, ec=colors[0], bins=5)
        plt.hist(data[i, label == 1], alpha=0.5, label=classes[1], density=True, ec=colors[1], bins=5)
        plt.title(f'Feature {i+1}')
        plt.legend(loc='best')
    plt.show()

def main():
    data, label = load_iris_binary()
    
    data_train, label_train, data_test, label_test = split_data(data=data, label=label)
    
    lda_train = lda(data=data_train, label=label_train, n_features=1)
    plot_iris_histogram(data=lda_train, label=label_train)
    lda_test = lda(data=data_test, label=label_test, n_features=1)
    plot_iris_histogram(data=lda_test, label=label_test)
    
    
    
if __name__ == "__main__":
    main()