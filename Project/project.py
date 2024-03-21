import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data_matrix = np.loadtxt('trainData.txt', delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt('trainData.txt', delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

def plot_histogram(data, label):
    # Plot histograms for dataset features
    plt.figure()
    classes = ["Fake", "Real"]
    colors = ["blue", "orange"]
    
    for i in range(data.shape[0]):
        plt.subplot(2, 3, i+1)
        for j in range(2):
            plt.hist(data[i, label == j], alpha=0.5, label=classes[j], density=True, ec=colors[j])
        plt.title(f'Feature {i+1}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_scatter(data, label):
    # Plot scatter plots for dataset features
    plt.figure(figsize=(10, 10))
    classes = ["Fake", "Real"]
    colors = ["blue", "orange"]
    
    x, _ = data.shape
    
    for i in range(x):
        for j in range(x):
            if i != j:
                plt.subplot(x, x, i*x + j + 1)
                for k in range(2):
                    plt.scatter(data[j, label == k], data[i, label == k], label=classes[k], s=1, c=colors[k])
                plt.xlabel(f'Feature {j+1}')
                plt.ylabel(f'Feature {i+1}')
                plt.legend(loc='best')
            else:
                plt.subplot(x, x, i*x + j + 1)
                for k in range(2):
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
    data, label = load_data()
    # print(f'data shape: {data.shape}')
    # print(f'label shape: {label.shape}')
    plot_histogram(data, label)
    plot_scatter(data, label)
    compute_statistics(data)
    
if __name__ == "__main__":
    main()