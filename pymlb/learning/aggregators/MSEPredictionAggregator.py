from typing import List
import numpy as np
import numpy.linalg as la
from pymlb.learning.aggregators import PredictionAggregator


class MSEPredictionAggregator(PredictionAggregator):
    def __init__(self, regularization_mult: float = 1):
        self.regularization_mult = regularization_mult
        super().__init__()

    def aggregate_train(self, sample_weights: np.ndarray, predictions: List[np.ndarray], ground_truths: np.ndarray):
        weights = []
        for stat_index in range(predictions[0].shape[-1]):
            x = np.hstack([prediction[:, stat_index:stat_index + 1] for prediction in predictions])
            y = ground_truths[:, stat_index]

            # add in the X bias
            # x = np.hstack([x, np.ones((x.shape[0], 1))])

            if sample_weights is not None:
                # add in the sample weights
                y *= np.sqrt(sample_weights)
                x *= np.sqrt(np.reshape(sample_weights, (-1, 1)))

            # regularize it
            add_to_one_weight = 100
            add_to_one_row = np.ones((x.shape[1],)) * add_to_one_weight
            regularizer = np.identity(x.shape[1]) * len(predictions) * 2 * self.regularization_mult
            x = np.vstack([x, regularizer, add_to_one_row])
            y = np.concatenate([y, np.zeros((x.shape[1],)), np.ones((1,)) * add_to_one_weight])

            # solve the system
            weight_vector = la.lstsq(x, y)[0]
            weights.append(weight_vector)

        return np.array(weights)  # (stat_count) x (model_count) matrix, where all rows should sum to ~1

    def aggregate_predict(self, predictions: List[np.ndarray], training_results):
        # flatten the predictions
        prediction_matrix = np.transpose(np.array(predictions), axes=[1, 2, 0])

        # add in a bias
        # prediction_matrix = np.concatenate([prediction_matrix, np.ones(prediction_matrix.shape[:-1] + (1,))], axis=-1)

        # obtain the results
        return np.vstack(
            [np.expand_dims(np.sum(prediction_sample * training_results, axis=-1), axis=0) for prediction_sample in
             prediction_matrix])

    def aggregate_input_gradients(self, gradients: List[np.ndarray], output_name: str, feature_index: int,
                                  training_results):
        gradient_matrix = np.transpose(np.concatenate(gradients, axis=0),
                                       axes=[1, 2, 0])  # [timestep, input feature index, model]
        gradient_matrix *= np.reshape(training_results[feature_index, :], (1, 1, -1))
        return np.expand_dims(np.sum(gradient_matrix, axis=-1), axis=0)
