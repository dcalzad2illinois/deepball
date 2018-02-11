from typing import List, Optional
import numpy as np


class PredictionAggregator:
    def __init__(self):
        pass

    def aggregate_train(self, sample_weights: Optional[np.ndarray], predictions: List[np.ndarray], ground_truths: np.ndarray):
        # returns an object that contains the aggregation details that will be passed into aggregate_predict
        raise NotImplementedError

    def aggregate_predict(self, predictions: List[np.ndarray], training_results):
        raise NotImplementedError

    def aggregate_input_gradients(self, gradients: List[np.ndarray], output_name: str, feature_index: int, training_results):
        raise NotImplementedError
