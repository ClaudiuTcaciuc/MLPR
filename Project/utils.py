import numpy as np

def load_data():
    #TODO: make the path not hard coded
    path = './data/trainData.txt'
    
    data_matrix = np.loadtxt(path, delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt(path, delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

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

    # Variance
    var = np.var(data, axis=1).reshape(-1, 1)
    print(f'Variance\n {var}')
    
    # std
    std = np.std(data, axis=1).reshape(-1, 1)
    print(f'Standard deviation\n{std}')