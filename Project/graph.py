import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data, label, classes):
    # Plot histogram for the dataset features
    
    plt.figure()
    # TODO: make the color general
    colors = ["blue", "orange"]
    
    for i in range(data.shape[0]):
        plt.subplot(2, 3, i+1)
        for j in range(2):
            plt.hist(data[i, label == j], alpha=0.5, label=classes[j], density=True, ec=colors[j])
        plt.title(f'Feature {i+1}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_scatter(data, label, classes):
    # Plot scatter plots for dataset features
    
    plt.figure(figsize=(10, 10))
    # TODO: make the color general
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
