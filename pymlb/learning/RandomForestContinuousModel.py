from pymlb.data import SequenceMatrices
from pymlb.learning import Model
from typing import Dict
from os.path import isfile
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestContinuousModel(Model):
    def __init__(self, key_counts: Dict[str, int], file_name: str = None, **kwargs):
        super().__init__(key_counts=key_counts)

        self.model = None
        self.estimators = kwargs.pop("estimators", 20)
        self.preprocessing = {"interaction_order": kwargs.pop("interaction_order", 1),
                              "recursive_timestep_distance": kwargs.pop("recursive_timestep_distance", 1)}

        if file_name is not None and isfile(file_name):
            self.import_model(file_name)

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.model, file)

    def import_model(self, file_name: str):
        if isfile(file_name):
            with open(file_name, 'rb') as file:
                self.model = pickle.load(file)
            if "preprocessing" in self.model:
                self.preprocessing = self.model["preprocessing"]

    def train(self, matrices: SequenceMatrices, *args, **kwargs):
        self.model = {
            "preprocessing": self.preprocessing,
            "estimators": self.estimators,
            "outputs": {},
            "input_modification": {}
        }

        modified_matrices, sample_weights = matrices.flatten_all(missing_value_replacement=0, **self.preprocessing)

        # concatenate each set of inputs and outputs into one big input matrix
        input_matrix = np.concatenate(
            [matrix for key, matrix in sorted(modified_matrices.items()) if key.startswith("in_")], axis=-1)

        # # standardize the input values
        # input_means = np.mean(input_matrix, axis=0, keepdims=True)
        # self.model["input_modification"]["mean"] = input_means
        # input_stddevs = np.std(input_matrix, axis=0, keepdims=True) + 0.00001  # use this epsilon so we don't get NaNs
        # self.model["input_modification"]["stddev"] = input_stddevs
        # input_matrix = (input_matrix - input_means) / input_stddevs

        # create the model for each output entry
        model = {}
        for key, output_matrix in modified_matrices.items():
            if not key.startswith("out_"):
                continue

            model[key] = RandomForestRegressor(n_estimators=self.estimators, n_jobs=-1, max_features=0.333, verbose=1)
            model[key].fit(input_matrix, output_matrix, sample_weights[key])

        self.model["outputs"] = model

    def predict(self, matrices: SequenceMatrices, intermediate_layer: str = None, *args, **kwargs):
        if intermediate_layer is not None and intermediate_layer not in self.model["outputs"]:
            return {}

        time_dimension = matrices.get_max_sequence_length()
        modified_matrices, sample_weights = matrices.flatten_all(remove_zero_samples=False,
                                                                 missing_value_replacement=0, **self.preprocessing)

        # concatenate each set of inputs and outputs into one big input matrix
        input_matrix = np.concatenate(
            [matrix for key, matrix in sorted(modified_matrices.items()) if key.startswith("in_")], axis=-1)

        # evaluate the regression for each output
        results = {}
        for key, model in self.model["outputs"].items():
            if not key.startswith("out_") or (intermediate_layer is not None and key != intermediate_layer):
                continue

            results[key] = model.predict(input_matrix)

            # restore the time dimension
            if time_dimension is not None:
                results[key] = np.reshape(results[key], (-1, time_dimension, results[key].shape[-1]))

        return results

    def summary(self, *args, **kwargs):
        # for each layer and output stat, output the model weights
        pass

    def input_gradients(self, sample_data: Dict[str, np.ndarray], feature_index: int, timestep_index: int,
                        output_name: str):
        raise NotImplementedError

    def feature_gradients(self, sample_data: SequenceMatrices, layer_name: str, output_name: str):
        raise NotImplementedError
