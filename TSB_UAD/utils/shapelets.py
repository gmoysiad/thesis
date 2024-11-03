import numpy as np
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

def calculate_shapelet_similarity(dataset1, dataset2):
    '''Calculate the similarty of two time series by checking the similarity of subsections of the time series'''


    # Check lengths and pad the shorter one to make both the same length
    max_length = max(len(dataset1), len(dataset2))
    if len(dataset1) < max_length:
        dataset1 = np.pad(dataset1, (0, max_length - len(dataset1)), mode='constant')
    if len(dataset2) < max_length:
        dataset2 = np.pad(dataset2, (0, max_length - len(dataset2)), mode='constant')

    # Reshape datasets to 2D arrays (n_samples, n_timestamps)
    dataset1 = dataset1.reshape(1, -1)
    dataset2 = dataset2.reshape(1, -1)

    # Combine datasets to simulate a multi-class problem
    combined_data = np.vstack([dataset1, dataset2])
    synthetic_labels = np.array([0, 1])  # Synthetic labels for fitting the model

    # Define shapelet sizes based on dataset size
    n_shapelets = 5  # Total number of shapelets
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(
        n_ts=len(combined_data),        # Number of time series
        ts_sz=combined_data.shape[1],   # Time series length
        n_classes=2,                    # Assuming synthetic labels (2 classes)
        l=0.1,                          # Proportion of time series length for minimum shapelet length (10%)
        r=1                             # Proportion of time series length for maximum shapelet length (100%)
    )

    # Initialize ShapeletModel
    shp_model = ShapeletModel(
        n_shapelets_per_size=shapelet_sizes, 
        optimizer="sgd", 
        weight_regularizer=0.01, 
        max_iter=200
    )

    # Fit model on the combined data with synthetic labels
    shp_model.fit(combined_data, synthetic_labels)

    # Transform the datasets using the fitted model
    dataset1_shapelets = shp_model.transform(dataset1)  # .astype(np.float64)
    dataset2_shapelets = shp_model.transform(dataset2)  # .astype(np.float64)

    '''try:
        raw_similarity = pairwise_distances(dataset1_shapelets, dataset2_shapelets, metric='euclidean')[0][0]
        raw_similarity = np.nan_to_num(raw_similarity, nan=1e10, posinf=1e10, neginf=0)
    except ValueError:
        raw_similarity = 1e10  

    # Compute similarity (e.g., Euclidean distance between transformed shapelet representations)
    raw_similarity = pairwise_distances(dataset1_shapelets, dataset2_shapelets, metric='euclidean')[0][0]
    raw_similarity = np.nan_to_num(raw_similarity, nan=1e10, posinf=1e10, neginf=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform([[raw_similarity]]).flatten()[0]'''

    return min(np.sqrt(np.sum((dataset1_shapelets-dataset2_shapelets) ** 2)), 1.0)
    # return raw_similarity  # Closer to 0 means more similar


