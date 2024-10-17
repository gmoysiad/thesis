import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_dtw_distance(X, Y):
    """
    Compute the Dynamic Time Warping (DTW) distance between two datasets X an Y.
    This function assumes that X an Y are numpy arrays of shape (n_samples, n_features).

    Args:
        X (np.ndarray): First dataset in the form of a 1D array (e.g., array([ 0.32,  0.37,  0.48, ..., -0.04, -0.03, -0.08])).
        Y (np.ndarray): Second dataset in the form of a 1D array.

    Returns:
        float: The DTW score, where a lower score indicates more similarity between the two datasets.
    """

    # Convert 1D arrays to 2D arrays (required by rbf_kernel)
    X = np.asarray(X).reshape(-1, 1)  # Reshape to (n_samples, 1)
    Y = np.asarray(Y).reshape(-1, 1)

    distance, _ = fastdtw(X, Y, dist=euclidean)
    return distance