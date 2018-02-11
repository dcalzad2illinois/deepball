from typing import List
import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA
from pymlb.learning.aggregators import PredictionAggregator


class PCAPredictionAggregator(PredictionAggregator):
    def __init__(self, tanh_transform: bool = False, column_normalization_order=2):
        self.tanh_transform = tanh_transform
        self.tanh_epsilon = 1e-6
        self.column_normalization_order = column_normalization_order
        super().__init__()

    def aggregate_train(self, sample_weights: np.ndarray, predictions: List[np.ndarray], ground_truths: np.ndarray):
        # ground_truths are not supported here

        n_components = predictions[0].shape[-1]
        predictions = np.concatenate(predictions, axis=-1)

        # perform PCA
        pca = PCA(n_components)
        pca.fit(np.arctanh(predictions * (1 - self.tanh_epsilon)) if self.tanh_transform else predictions)

        # normalize the columns so |column|_1 = 1
        return pca.components_.T / la.norm(pca.components_.T, ord=self.column_normalization_order, axis=0,
                                           keepdims=True)

    def aggregate_predict(self, predictions: List[np.ndarray], training_results):
        predictions = np.concatenate(predictions, axis=-1)

        # use the weights to get this new transformation
        results = np.tanh(np.arctanh(predictions * (1 - self.tanh_epsilon)).dot(
            training_results)) if self.tanh_transform else predictions.dot(training_results)

        return results

    def aggregate_input_gradients(self, gradients: List[np.ndarray], output_name: str, feature_index: int, training_results):
        # TODO
        raise NotImplementedError
