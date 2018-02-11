from typing import List
import numpy as np
from pymlb.learning.aggregators import PredictionAggregator
from pymlb.learning.optimization import GaussianCombiner
from math import sqrt


class GaussianPredictionMerger(PredictionAggregator):
    def aggregate_train(self, sample_weights: np.ndarray, predictions: List[np.ndarray], ground_truths: np.ndarray):
        n = int((sqrt(4 * predictions[0].shape[-1] + 1) - 1) / 2)

        combiner = GaussianCombiner(n, len(predictions), iterations=100)
        return combiner.optimize(predictions, ground_truths)

    def aggregate_predict(self, predictions: List[np.ndarray], training_results):
        n = int((sqrt(4 * predictions[0].shape[-1] + 1) - 1) / 2)
        return GaussianCombiner(n, len(predictions)).evaluate(predictions, training_results)

    def aggregate_input_gradients(self, gradients: List[np.ndarray], output_name: str, feature_index: int,
                                  training_results):
        return gradients[training_results]
