import numpy as np
import matplotlib.pyplot as plt

def vRow(x):
    return x.reshape(1, -1)

def vCol(x):
    return x.reshape(-1, 1)

def logarithmic_gau_pdf(sample, class_mean, class_covariance):
    d = sample.shape[1]  # Number of features
    sample = sample - class_mean.reshape(1, -1)
    inv_cov = np.linalg.inv(class_covariance)
    sign, log_det = np.linalg.slogdet(class_covariance)
    det_sign = sign * log_det
    log_pdf = -0.5 * d * np.log(2 * np.pi) - 0.5 * det_sign - 0.5 * np.dot(np.dot(sample, inv_cov), sample.T).diagonal()
    return log_pdf

def main():
    # PART 1
    plt.figure()
    Xplot = np.linspace(-8, 12, 1000).reshape(-1, 1)  # Reshape to a column vector
    m = np.array([[1.0]])  # Mean vector
    C = np.array([[2.0]])  # Covariance matrix
    plt.plot(Xplot.ravel(), np.exp(logarithmic_gau_pdf(Xplot, m, C)))
    plt.show()
    
    pdfSol = np.load('Solution/llGAU.npy')
    pdfGau = logarithmic_gau_pdf(Xplot, m, C)
    print(f"max diff: {np.max(np.abs(pdfSol - pdfGau))}")
    
    # PART 2
    data = np.load('Solution/XND.npy').T
    
    mean = np.mean(data, axis=0).reshape(-1, 1)
    print(f"mean: \n{mean}")
    centered_data = data - mean.reshape(1, -1)
    cov = np.dot(centered_data.T, centered_data) / data.shape[0]
    print(f"cov: \n{cov}")
    
    pdfGau  = logarithmic_gau_pdf(data, mean, cov)
    print(f"pdfGau: \n{np.sum(pdfGau)}")
    
    # PART 3
    data = np.load('Solution/X1D.npy').T
    
    mean = np.mean(data, axis=0).reshape(-1, 1)
    print(f"mean: \n{mean}")
    centered_data = data - mean.reshape(1, -1)
    cov = np.dot(centered_data.T, centered_data) / data.shape[0]
    print(f"cov: \n{cov}")
    
    pdfGau  = logarithmic_gau_pdf(data, mean, cov)
    print(f"pdfGau: \n{np.sum(pdfGau)}")
    
    plt.figure()
    plt.hist(data, bins=50, density=True, ec='black')
    plt.plot(Xplot.ravel(), np.exp(logarithmic_gau_pdf(Xplot, mean, cov)))
    plt.show()

if __name__ == "__main__":
    main()