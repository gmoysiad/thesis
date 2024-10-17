import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# Maximum Mean Discrepancy function
def calculate_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets X an Y.
    This function assumes that X an Y are numpy arrays of shape (n_samples, n_features).

    Args:
        X (np.ndarray): First dataset in the form of a 1D array (e.g., array([ 0.32,  0.37,  0.48, ..., -0.04, -0.03, -0.08])).
        Y (np.ndarray): Second dataset in the form of a 1D array.
        gamma (float): Parameter for the RBF kernel (default is 1.0).

    Returns:
        float: The MMD score, where a lower score indicates more similarity between the two datasets.
    """

    # Convert 1D arrays to 2D arrays (required by rbf_kernel)
    X = np.asarray(X).reshape(-1, 1)  # Reshape to (n_samples, 1)
    Y = np.asarray(Y).reshape(-1, 1)

    # Compute kernel matrices using RBF kernel
    K_XX = rbf_kernel(X, X, gamma=gamma)  # Kernel matrix for X
    K_YY = rbf_kernel(Y, Y, gamma=gamma)  # Kernel matrix for Y
    K_XY = rbf_kernel(X, Y, gamma=gamma)  # Kernel matrix between X and Y

    # Calculate MMD using the kernel matrices
    m = len(X)
    n = len(Y)
    
    # Mean of K_XX, K_YY, and K_XY
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    
    return mmd
