import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# Maximum Mean Discrepancy function
def calculate_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets X an Y.
    This function assumes that X an Y are numpy arrays of shape (n_samples, n_features).

    Parameters:
    - X: numpy array of shape (n_samples_X, n_features)
    - Y: numpy array of shape (n_samples_Y, n_features)
    - kernel: Kernel function to use. For now, only 'rbf' (radical Basis Function) is implemented.
    - gamma: Parameter for the RBF kernel.

    Returns:
    - mmd: A scalar value representing the similarity (lower means more similar).
    """

    # Compute kernel matrices
    if kernel == 'rbf':
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
    
    # MMD formula
    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    
    return mmd