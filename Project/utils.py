import numpy as np
import scipy

def load_data():
    #TODO: make the path not hard coded
    path = './data/trainData.txt'
    
    data_matrix = np.loadtxt(path, delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt(path, delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

def compute_statistics(data):
    """ Compute the mean, variance, std and covariance matrix of the data """
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
    

def pca(data, n_features=4):
    """ Compute the PCA decomposition on n features
        Data: (n_features, n_samples)
    """
    
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
    """ Compute the within-class and between
            Sw: within-class scatter matrix
                formula: sum(Cov(Xi) for i in classes)
            Sb: between-class scatter matrix
                formula: sum(Ni * (mean(Xi) - mean(X)) for i in classes)
    """
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
    """ Compute the LDA decomposition on n features
            n_max = n_features - 1
        Data: (n_features, n_samples)
    """
    n, _ = data.shape
    
    if n_features > n:
        raise ValueError(f"n_features must be less than {n}")
    
    Sw, Sb = compute_Sw_Sb(data=data, label=label)
    
    _, eigen_vectors = scipy.linalg.eigh(Sb, Sw)
    selected_eigen_vectors = eigen_vectors[:, ::-1][:, :n_features]
    lda_data = np.dot(selected_eigen_vectors.T, data)
    
    return lda_data
