from typing import List
import numpy as np
import numpy.linalg as la
from pymlb.learning.aggregators import PredictionAggregator
from math import sqrt, pi, log


class GaussianPredictionAggregator(PredictionAggregator):
    def aggregate_train(self, sample_weights: np.ndarray, predictions: List[np.ndarray], ground_truths: np.ndarray):
        # our goal will be to find the ideal mixture of the provided gaussians to approximate this
        n = int((sqrt(4 * predictions[0].shape[-1] + 1) - 1) / 2)

        # extract the mean vectors from the given predictions
        means = [prediction[:, :n] for prediction in predictions]

        # extract the covariance matrices from the given predictions
        precisions = [np.reshape(prediction[:, n:], (prediction.shape[0], n, n)) for prediction in predictions]

        # create the gaussian log likelihood function
        def log_likelihood(x, u, covariance_inverse):
            det_covariance_inverse = la.det(covariance_inverse)
            return 0.5 * (-len(x) * log(2 * pi) + log(det_covariance_inverse) - np.reshape(x - u, (1, -1)).dot(
                covariance_inverse).dot(np.reshape(x - u, (-1, 1))))

        # find the likelihoods on the validation set
        likelihoods = []
        for mean, precision, prediction in zip(means, precisions, predictions):
            likelihoods.append(sum(
                log_likelihood(ground_truths[row_index, :mean.shape[1]], mean[row_index, :], precision[row_index, :, :]) *
                sample_weights[row_index] for row_index in range(prediction.shape[0])))

        # save the index for the best system
        return np.argmax(np.array(likelihoods))

    def aggregate_predict(self, predictions: List[np.ndarray], training_results):
        return predictions[training_results]

    def aggregate_input_gradients(self, gradients: List[np.ndarray], output_name: str, feature_index: int,
                                  training_results):
        return gradients[training_results]
