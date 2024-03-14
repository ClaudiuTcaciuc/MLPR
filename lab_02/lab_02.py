import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


# load binary iris dataset
def load_iris_binary():
    data, label = datasets.load_iris()['data'].T, datasets.load_iris()['target']
    return data, label

def plot_iris_histogram(data, label):
    plt.figure(figsize=(15, 12))
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.hist(data[i, label == 0], alpha=0.5, label='setosa', density=True, ec='blue')
        plt.hist(data[i, label == 1], alpha=0.5, label='versicolor', density=True, ec='orange')
        plt.hist(data[i, label == 2], alpha=0.5, label='virginica', density=True, ec='green')
        plt.title(f'Feature {i+1}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_scatter(data, label):
    plt.figure(figsize=(10, 10))
    
    data_c1 = data[:, label == 0]
    data_c2 = data[:, label == 1]
    data_c3 = data[:, label == 2]
    
    for i in range(4):
        for j in range(4):
            if i != j:
                plt.subplot(4, 4, i*4 + j + 1)
                plt.scatter(data_c1[i], data_c1[j], label='setosa', s=10)
                plt.scatter(data_c2[i], data_c2[j], label='versicolor', s=10)
                plt.scatter(data_c3[i], data_c3[j], label='virginica', s=10)
                plt.xlabel(f'Feature {i+1}')
                plt.ylabel(f'Feature {j+1}')
                plt.legend(loc='best')
            else: # not needed, but cool
                plt.subplot(4, 4, i*4 + j + 1)
                plt.hist(data[i, label == 0], alpha=0.5, label='setosa', density=True, ec='blue')
                plt.hist(data[i, label == 1], alpha=0.5, label='versicolor', density=True, ec='orange')
                plt.hist(data[i, label == 2], alpha=0.5, label='virginica', density=True, ec='green')
                plt.legend(loc='best')
                plt.ylabel(f'Feature {i+1}')
    plt.tight_layout()
    plt.show()

def main():
    data, label = load_iris_binary()
    plot_iris_histogram(data, label)
    plot_scatter(data, label)

if __name__ == "__main__":
    main()
