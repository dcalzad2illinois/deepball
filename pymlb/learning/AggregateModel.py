from pymlb.data import SequenceMatrices
from pymlb.learning import Model
from pymlb.learning.aggregators import PredictionAggregator
from typing import Dict
from os.path import isfile
import pickle
import numpy as np


class AggregateModel(Model):
    def __init__(self, template: Model, key_classes: Dict[str, PredictionAggregator], file_name: str = None):
        self.template = template
        self.key_aggregators = key_classes  # TODO: check that each value inherits from PredictionAggregator
        self.model = {"aggregators": {}, "files": []}
        super().__init__(key_counts=template.get_key_counts())

        if file_name is not None and isfile(file_name):
            self.import_model(file_name)

    def is_trained(self):
        return len(self.model["files"]) > 0

    def __load_sub_model(self, file_name: str):
        self.template.import_model(file_name)

    def __sub_model_predict(self, matrices: SequenceMatrices, intermediate_layer: str = None):
        return self.template.predict(matrices, intermediate_layer=intermediate_layer)

    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.model, file)

    def import_model(self, file_name: str):
        if isfile(file_name):
            with open(file_name, 'rb') as file:
                self.model = pickle.load(file)

    @staticmethod
    def _get_flattened_data(matrices: SequenceMatrices, predictions, key):
        # this is a key that we need to aggregate. obtain the weights, predictions, and ground truths
        sample_weights = (None if key not in matrices.get_key_counts() else matrices.get_sample_weights_for(key, len(
            matrices.get_key_counts()[key]) == 2))
        if sample_weights is None:
            sample_weights = matrices.get_default_sample_weights()
        these_predictions = [prediction[key] for prediction in predictions]
        ground_truths = None if key not in matrices.get_key_counts() else matrices.get_key_matrices()[key]

        # flatten everything
        original_shape = sample_weights.shape
        these_predictions = [np.reshape(prediction, (-1, prediction.shape[-1])) for prediction in these_predictions]
        sample_weights = np.reshape(sample_weights, (-1,))
        if ground_truths is not None:
            ground_truths = np.reshape(ground_truths, (-1, ground_truths.shape[-1]))

        return sample_weights, these_predictions, ground_truths, original_shape

    def train(self, matrices: SequenceMatrices, *args, **kwargs):
        self.model["files"] = list(sorted(kwargs.get("files")))

        # get the predictions for each sub-model
        predictions = []
        for model_file_name in self.model["files"]:
            self.__load_sub_model(model_file_name)
            this_model_predictions = self.__sub_model_predict(matrices)
            for key in self.key_aggregators:
                if key not in this_model_predictions:
                    sub_predictions = self.__sub_model_predict(matrices, intermediate_layer=key)
                    if key in sub_predictions:
                        this_model_predictions[key] = sub_predictions[key]
            predictions.append(this_model_predictions)

        for key in self.key_aggregators:
            sample_weights, these_predictions, ground_truths, _ = self._get_flattened_data(matrices, predictions, key)

            # filter out only the predictions where sample_weights > 0, and do the same with the rest
            these_predictions = [prediction[np.argwhere(sample_weights > 0)][:, 0, :] for prediction in
                                 these_predictions]
            if ground_truths is not None:
                ground_truths = ground_truths[np.argwhere(sample_weights > 0)][:, 0, :]
            sample_weights = sample_weights[np.argwhere(sample_weights > 0)][:, 0]

            # train this aggregator
            self.model["aggregators"][key] = self.key_aggregators[key].aggregate_train(sample_weights,
                                                                                       these_predictions,
                                                                                       ground_truths)

    def predict(self, matrices: SequenceMatrices, intermediate_layer: str = None, *args, **kwargs):
        # get the predictions for each sub-model
        predictions = []
        for model_file_name in self.model["files"]:
            self.__load_sub_model(model_file_name)
            predictions.append(self.__sub_model_predict(matrices, intermediate_layer=intermediate_layer))

        output_predictions = {}
        for key in self.key_aggregators.keys():
            if key not in predictions[0] or (intermediate_layer is not None and key != intermediate_layer):
                continue

            # this is a key that we need to predict the output of

            # obtain the results of the aggregator on the flattened version
            _, these_predictions, _, original_shape = self._get_flattened_data(matrices, predictions, key)
            aggregate_predictions = self.key_aggregators[key].aggregate_predict(these_predictions,
                                                                                self.model["aggregators"][key])

            # reshape it to the original dimensions and save it
            aggregate_predictions = np.reshape(aggregate_predictions, original_shape + (-1,))
            output_predictions[key] = aggregate_predictions

        return output_predictions

    def summary(self, *args, **kwargs):
        # for each layer and output stat, output the model weights
        # temporary - just print the summary of the template
        self.template.summary()

    def input_gradients(self, sample_data: Dict[str, np.ndarray], feature_index: int, timestep_index: int,
                        output_name: str):

        # find the gradients of each sub-model
        sub_model_gradients = []
        for model_file_name in self.model["files"]:
            self.__load_sub_model(model_file_name)
            sub_model_gradients.append(
                self.template.input_gradients(sample_data=sample_data, feature_index=feature_index,
                                              timestep_index=timestep_index, output_name=output_name))

        # weight these gradients by each model's weight for this specific output_name (e.g. out_stats) and feature_index
        gradients = {}
        for key in sub_model_gradients[0].keys():
            gradients[key] = self.key_aggregators[output_name].aggregate_input_gradients(
                [sub_gradient[key] for sub_gradient in sub_model_gradients], output_name=output_name,
                feature_index=feature_index, training_results=self.model["aggregators"][output_name])

        return gradients

    def feature_gradients(self, sample_data: SequenceMatrices, layer_name: str, output_name: str):
        raise NotImplementedError
