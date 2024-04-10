import numpy as np
import matplotlib.pyplot as plt

class multivariate_gaussian_classifier:
    def __init__(self, data) -> None:
        data = data
        mean = []
        covariance = []
        score = []
    
    def __logaritmic_gau_pdf__(self, sample, class_mean, class_covariance):
        d = sample.shape[0]
        sample = sample - class_mean
        inv_cov = np.linalg.inv(class_covariance)
        log_pdf = -0.5 * np.dot(np.dot(sample.T, inv_cov), sample) - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(class_covariance))
        return log_pdf

    def __compute_gau_score__(self):
        score = []
        for i in range(len(self.mean)):
            score.append(self.__logaritmic_gau_pdf__(self.data, self.mean[i], self.covariance[i]))
        return score
    
    def run(self):
        unique_label = np.unique(self.label)
        for i in unique_label:
            self.mean.append(np.mean(self.data[:, self.label==i], axis=1).reshape(-1, 1))
            self.covariance.append(np.cov(self.data[:, self.label==i]))
        self.score = self.__compute_gau_score__()

def vRow(x):
    return x.reshape(1, x.size)

def vCol(x):
    return x.reshape(x.size, 1)

def main():
    data = np.load("Solution/XND.npy").T

if __name__ == "__main__":
    main()
