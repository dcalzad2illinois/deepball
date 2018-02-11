from typing import Dict
from pymlb.data import SequenceMatrices
import numpy as np


class Model:
    def __init__(self, key_counts: Dict[str, int] = None):
        self.key_counts = key_counts

    def save(self, file_name):
        raise NotImplementedError

    def import_model(self, file_name: str):
        raise NotImplementedError

    def train(self, matrices: SequenceMatrices, *args, **kwargs):
        raise NotImplementedError

    def predict(self, matrices: SequenceMatrices, intermediate_layer: str = None, *args, **kwargs):
        raise NotImplementedError

    def summary(self, *args, **kwargs):
        raise NotImplementedError

    def input_gradients(self, sample_data: Dict[str, np.ndarray], feature_index: int, timestep_index: int,
                        output_name: str):
        raise NotImplementedError

    def feature_gradients(self, sample_data: SequenceMatrices, layer_name: str, output_layer: str):
        raise NotImplementedError

    def get_key_counts(self):
        return self.key_counts

    def visualize(self, to_file: str, *args, **kwargs):
        raise NotImplementedError
