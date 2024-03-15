import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def load_iris_binary():
    # Load binary iris dataset
    iris_data = datasets.load_iris()
    data, label = iris_data['data'].T, iris_data['target']
    return data, label

def plot_iris_histogram(data, label):
    # Plot histograms for iris dataset features
    plt.figure(figsize=(15, 12))
    classes = ['setosa', 'versicolor', 'virginica']
    colors = ['blue', 'orange', 'green']
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        for j in range(3):
            plt.hist(data[i, label == j], alpha=0.5, label=classes[j], density=True, ec=colors[j])
        plt.title(f'Feature {i+1}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_iris_scatter(data, label):
    # Plot scatter plots for iris dataset features
    plt.figure(figsize=(10, 10))
    classes = ['setosa', 'versicolor', 'virginica']
    colors = ['blue', 'orange', 'green']
    
    for i in range(4):
        for j in range(4):
            if i != j:
                plt.subplot(4, 4, i*4 + j + 1)
                for k in range(3):
                    plt.scatter(data[j, label == k], data[i, label == k], label=classes[k], s=10, c=colors[k])
                plt.xlabel(f'Feature {j+1}')
                plt.ylabel(f'Feature {i+1}')
                plt.legend(loc='best')
            else:
                plt.subplot(4, 4, i*4 + j + 1)
                for k in range(3):
                    plt.hist(data[i, label == k], alpha=0.5, label=classes[k], density=True, ec=colors[k])
                plt.legend(loc='best')
                plt.ylabel(f'Feature {i+1}')
    plt.tight_layout()
    plt.show()

def compute_statistics(data):
    # Empirical dataset mean
    mu_class = np.mean(data, axis=1).reshape(-1, 1)
    print(f'Empirical dataset mean\n{mu_class}')
    
    # Centered data
    centered_data = data - mu_class
    print(f'Centered data shape\n{centered_data.shape}')
    
    # Covariance matrix
    cov_matrix = np.cov(centered_data)
    print(f'Covariance matrix shape\n{cov_matrix.shape}')
    # C = (centered_data @ centered_data.T) / (centered_data.shape[1] - 1)
    
    # Variance
    var = np.var(data, axis=1).reshape(-1, 1)
    print(f'Variance\n {var}')
    
    # std
    std = np.std(data, axis=1).reshape(-1, 1)
    print(f'Standard deviation\n{std}')

def main():
    data, label = load_iris_binary()
    plot_iris_histogram(data, label)
    plot_iris_scatter(data, label)
    compute_statistics(data)

if __name__ == "__main__":
    main()
