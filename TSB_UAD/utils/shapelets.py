import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.utils import check_random_state

def sliding_window_view(arr, window_size, step_size=1):
    """Helper function to get sliding windows of the array."""
    n = arr.shape[0]
    return np.array([arr[i:i + window_size] for i in range(0, n - window_size + 1, step_size)])

def shapelet_distance(shapelet, series):
    """
    Calculate the minimum distance between a shapelet and a time series by sliding the shapelet over the series.
    
    Parameters:
    - shapelet: numpy array, shapelet subsequence
    - series: numpy array, full time series
    
    Returns:
    - min_dist: minimum Euclidean distance between the shapelet and the sliding windows of the series
    """
    windows = sliding_window_view(series, len(shapelet))
    dists = np.array([euclidean(shapelet, window) for window in windows])
    return np.min(dists)

def extract_shapelets(dataset, shapelet_size, num_shapelets, random_state=None):
    """
    Extract random shapelets from the dataset.
    
    Parameters:
    - dataset: numpy array of shape (n_samples, series_length)
    - shapelet_size: int, the size of the shapelets to extract
    - num_shapelets: int, the number of shapelets to extract
    - random_state: random state for reproducibility
    
    Returns:
    - shapelets: list of numpy arrays representing shapelets
    """
    rng = check_random_state(random_state)

    dataset = np.asarray(dataset).reshape(-1, 1)  # Reshape to (n_samples, 1)

    n_samples, series_length = dataset.shape
    shapelets = []
    
    for _ in range(num_shapelets):
        # Randomly select a time series and a start position for the shapelet
        ts_idx = rng.randint(n_samples)
        start_pos = rng.randint(0, series_length - shapelet_size)
        shapelet = dataset[ts_idx, start_pos:start_pos + shapelet_size]
        shapelets.append(shapelet)
    
    return shapelets

def calculate_shapelet_similarity(shapelets1, shapelets2):
    """
    Calculate the average distance between shapelets from two datasets.

    By comparing the similarity of the extracted shapelets, we get an estimate of how similare two datasets
    are. Lower shapelet similarity suggests the datasets have similar underlying patterns.
    
    Parameters:
    - shapelets1: list of shapelets from the first dataset
    - shapelets2: list of shapelets from the second dataset
    
    Returns:
    - avg_dist: average minimum distance between shapelets from the two datasets
    """
    total_dist = 0
    count = 0
    for s1 in shapelets1:
        for s2 in shapelets2:
            dist = euclidean(s1, s2)
            total_dist += dist
            count += 1
    return total_dist / count if count > 0 else np.inf

# Example usage

# Create synthetic time series datasets
# np.random.seed(42)
# dataset1 = np.random.normal(0, 1, (50, 100))  # 50 samples, each 100 time steps
# dataset2 = np.random.normal(0.5, 1, (50, 100))  # Another dataset with a shifted mean
# dataset3 = np.random.normal(0, 1, (50, 100))  # Similar to dataset1
# 
# # Extract shapelets from each dataset
# shapelet_size = 10
# num_shapelets = 5
# 
# shapelets1 = extract_shapelets(dataset1, shapelet_size, num_shapelets, random_state=42)
# shapelets2 = extract_shapelets(dataset2, shapelet_size, num_shapelets, random_state=42)
# shapelets3 = extract_shapelets(dataset3, shapelet_size, num_shapelets, random_state=42)
# 
# # Calculate shapelet-based similarity between dataset1 and dataset2 (different)
# shapelet_similarity_1_2 = calculate_shapelet_similarity(shapelets1, shapelets2)
# print(f"Shapelet similarity between Dataset 1 and Dataset 2: {shapelet_similarity_1_2}")
# 
# # Calculate shapelet-based similarity between dataset1 and dataset3 (similar)
# shapelet_similarity_1_3 = calculate_shapelet_similarity(shapelets1, shapelets3)
# print(f"Shapelet similarity between Dataset 1 and Dataset 3: {shapelet_similarity_1_3}")
