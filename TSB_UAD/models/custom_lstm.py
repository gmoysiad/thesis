import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.callbacks import EarlyStopping


import numpy as np
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler


class CustomLstm:
    def __init__(self, slidingwindow=100, predict_time_steps=1, neurons=50, contamination=0.1, epochs=10, patience=10,
                 verbose=0):
        super(CustomLstm, self).__init__()
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.neurons = neurons
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.model_name = 'Custom AutoEncoder LSTM'
        self.X_train_ = None
        self.Y = None
        self.estimation = None
        self.estimator = None
        self.n_initial = None
        self.n_test_ = None
        self.decision_scores_ = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.es = None

    def fit(self, X_clean, X_dirty, ratio=0.15):
        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)

        self.X_train, self.y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        self.X_test, self.y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(LSTM(self.neurons, return_sequences=True))
        self.model.add(LSTM(self.neurons // 2, return_sequences=False))
        self.model.add(tf.keras.layers.RepeatVector(self.X_train.shape[1]))
        self.model.add(LSTM(self.neurons // 2, return_sequences=True))
        self.model.add(LSTM(self.neurons, return_sequences=True))
        # self.model.add(Dense(predict_time_steps))
        self.model.add(tf.keras.layers.TimeDistributed(Dense(predict_time_steps)))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        
        self.model.fit(self.X_train, self.y_train, validation_split=ratio, epochs=self.epochs, batch_size=64,
                       verbose=self.verbose, callbacks=[self.es])
        
        prediction = self.model.predict(self.X_test)

        self.Y = self.y_test
        self.estimation = prediction
        self.estimator = self.model
        self.n_initial = self.X_train.shape[0]
        
        return self

    def create_dataset(self, X, slidingwindow, predict_time_steps=1): 
        Xs, ys = [], []
        for i in range(len(X) - slidingwindow - predict_time_steps+1):
            tmp = X[i: i + slidingwindow + predict_time_steps]
            tmp = MinMaxScaler(feature_range=(0, 1)).fit_transform(tmp.reshape(-1, 1)).ravel()
            x = tmp[:slidingwindow]
            y = tmp[slidingwindow:]
            
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)

    def decision_function(self, X=False, measure=None):
        """Derive the decision score based on the given distance measure
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        measure : object
            object for given distance measure with methods to derive the score
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if type(X) != bool:
            self.X_train_ = X
        n_test_ = self.n_test_
        Y_test = self.Y

        score = np.zeros(n_test_)
        estimation = self.estimation

        for i in range(estimation.shape[0]):
            score[i - estimation.shape[0]] = measure.measure(Y_test[i], estimation[i], n_test_ - estimation.shape[0] + i)

        score[0: - estimation.shape[0]] = score[- estimation.shape[0]]
        
        self.decision_scores_ = score
        return self
    
    def prediction(self, X_clean, X_dirty):
        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)

        self.X_train, self.y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        self.X_test, self.y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        
        prediction = self.model.predict(self.X_test)

        self.Y = self.y_test
        self.estimation = prediction
        self.estimator = self.model
        self.n_initial = self.X_train.shape[0]
        
        return self

    def predict_proba(self, X, method='linear', measure=None):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:
        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.
        measure: None
        Returns
        -------
        outlier_probability : numpy array of shape (n_samples,)
            For each observation, tells whether it should be considered
            as an outlier according to the fitted model. Return the
            outlier probability, ranging in [0,1].
        """

        # check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        self.fit(X)
        self.decision_function(measure=measure)
        test_scores = self.decision_scores_

        probs = np.zeros([X.shape[0], int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = math.erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')
