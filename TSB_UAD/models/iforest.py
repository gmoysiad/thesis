"""IsolationForest Outlier Detector. Implemented on scikit-learn library.
"""
# Author: Yinchen Wu <Yinchen@uchicago.edu>

from __future__ import division
from __future__ import print_function

from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

from ..utils.utility import invert_order
from .detectorB import DetectorB


class IForest(DetectorB):
    """Wrapper of scikit-learn IsolationForest Class with more functionalities.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.
    max_samples : int, float or string, optional, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.
    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
            - this attribute is useless of a sensor timeseries anomly

    bootstrap : bool, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.
    max_samples_ : integer
        The actual number of samples
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
                
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.model_name = 'IForest'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples (time series length). n_features corresponds to the subsequence length.
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        try: 
            X = check_array(X)
        except:
            X = X.reshape(-1, 1)
        
        X = check_array(X)
            
        self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                            max_samples=self.max_samples,
                                            contamination=self.contamination,
                                            max_features=self.max_features,
                                            bootstrap=self.bootstrap,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state,
                                            verbose=self.verbose)


        self.detector_.fit(X=X, y=None, sample_weight=None)

        # invert decision_scores_. Outliers comes with higher outlier scores.
        self.decision_scores_ = -self.detector_.score_samples(X)
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        # invert outlier scores. Outliers comes with higher outlier scores
        return invert_order(self.detector_.decision_function(X))



    def estimators_(self):
        """The collection of fitted sub-estimators.
        Decorator for scikit-learn Isolation Forest attributes.

        :meta private:
        """
        return self.detector_.estimators_


    def estimators_samples_(self):
        """The subset of drawn samples (i.e., the in-bag samples) for
        each base estimator.
        Decorator for scikit-learn Isolation Forest attributes.

        :meta private:
        """
        return self.detector_.estimators_samples_

    def max_samples_(self):
        """The actual number of samples.
        Decorator for scikit-learn Isolation Forest attributes.

        :meta private:
        """
        return self.detector_.max_samples_
