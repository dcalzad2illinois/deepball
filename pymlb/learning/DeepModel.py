import keras.backend as K
from typing import Dict, Union
from keras.models import Model as KerasModel, load_model
from pymlb.data import SequenceMatrices
from pymlb.learning import Model
from pymlb.learning.layers import *
from keras.layers import Lambda, Dropout, Highway, Dense, BatchNormalization, TimeDistributed, LSTM, Activation
from keras.regularizers import l1_l2
import tensorflow as tf
from os.path import isfile
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pymlb.learning import ModelCheckpointParallel
import numpy as np
import numpy.linalg as la
from pymlb.learning.custom_objectives import outlier_loss, percentile_loss, mae_variance, mse_relative, mae_relative, \
    gaussian_loss
from pymlb.learning.custom_activations import gaussian
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model


class DeepModel(Model):
    def __init__(self, file_name: str = None, key_counts: Dict[str, int] = None, reuse_structure: bool = True,
                 reg_l1=0., reg_l2=0., multi_gpu: bool = False):

        self.key_counts = key_counts
        self.multi_gpu = multi_gpu
        self.model_checkpoint_class = ModelCheckpoint
        self.batch_mult = 1

        # sift through each possible parameter combination so we end up with self.model filled
        if key_counts is not None and file_name is not None and isfile(file_name) and not reuse_structure:
            self.model = self.__wrap_gpu_model(self._construct_new(l1_l2(reg_l1, reg_l2)))
            self.model.load_weights(file_name, by_name=True)
            self.compile()
        elif key_counts is not None and (file_name is None or not isfile(file_name)):
            self.model = self.__wrap_gpu_model(self._construct_new(l1_l2(reg_l1, reg_l2)))
            self.compile()
        elif file_name is not None and isfile(file_name):
            self.model = self.__wrap_gpu_model(load_model(file_name, custom_objects=self.get_custom_objects()))
        else:
            raise RuntimeError("No valid parameter combination was found for DeepModel.__init__.")

        super().__init__(key_counts=key_counts)

    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    @staticmethod
    def __get_available_gpus():
        return [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

    def __wrap_gpu_model(self, model: KerasModel):
        # print(self.__get_available_gpus())
        # exit()
        if self.multi_gpu:
            gpu_count = len(self.__get_available_gpus())
            if gpu_count > 1:
                self.model_checkpoint_class = ModelCheckpointParallel
                self.batch_mult = gpu_count
                return multi_gpu_model(model, gpus=gpu_count)
        return model

    def save(self, file_name):
        self.model.save(file_name)

    @staticmethod
    def get_custom_objects():
        return {"NewBatchNormalization": NewBatchNormalization,
                "outlier_loss": outlier_loss,
                "percentile_loss": percentile_loss,
                "mae_variance": mae_variance,
                "mse_relative": mse_relative,
                "mae_relative": mae_relative,
                "gaussian_loss": gaussian_loss,
                "gaussian": gaussian}

    def import_model(self, file_name: str):
        self.model.load_weights(file_name, by_name=True)

    def __filter_unused_keys(self, dictionary):
        for output_layer in self.model.output_layers:
            layer_name = output_layer.name
            this_key = self._layer_name_to_key(layer_name)
            if this_key in dictionary:
                dictionary[layer_name] = dictionary[this_key]
        for k in list(dictionary.keys()):
            if not self.contains_layer(k):
                dictionary.pop(k)
        return dictionary

    def _no_flag(self, tensor, name: str = None):
        if tensor.get_shape()[-1] > 1:
            return TimeDistributed(Lambda(name=name, function=lambda x: x[:, 1:],
                                          output_shape=lambda x: (x[0], x[1] - 1)))(tensor)
        else:
            return tensor

    def _add_skip_layers(self, tensor_list, layer_count, activation="relu", regularization=None):
        for layer_index in range(layer_count):
            new_list = []
            for i, item in enumerate(tensor_list):
                item = TimeDistributed(
                    Highway(activation=activation, W_regularizer=regularization, b_regularizer=regularization))(item)
                new_list.append(item)
            tensor_list = new_list

        return tensor_list

    def _add_dense_layers(self, tensor_list, layer_count=1, activation="relu",
                          dimension_multiplier: float = 1, out_dim: int = None, regularization=None, names=None):

        for layer_index in range(layer_count):
            new_list = []
            for i, item in enumerate(tensor_list):
                name = None
                if names is not None:
                    name = names(layer_index, i)
                item = Activation(activation)(TimeDistributed(name=name,
                                                              layer=Dense(out_dim if out_dim is not None else int(
                                                                  int(item.get_shape()[-1]) * dimension_multiplier),
                                                                          kernel_regularizer=regularization,
                                                                          bias_regularizer=regularization))(item))
                new_list.append(item)
            tensor_list = new_list

        return tensor_list

    def _add_dropout_layers(self, tensor_list, p: float = 0.3):
        return [Dropout(p)(item) for item in tensor_list]

    def _add_lstm_layers(self, tensor_list, activation='tanh', u_drop_ratio=0.5, regularization=None,
                         out_dimension: int = None, kind=LSTM):

        # lstm for their last year stats
        lstms = [
            kind(out_dimension if out_dimension is not None else int(item.get_shape()[-1]), activation=activation,
                 return_sequences=True, recurrent_dropout=u_drop_ratio, kernel_regularizer=regularization,
                 bias_regularizer=regularization, recurrent_regularizer=regularization)(item)
            for item in tensor_list]

        return lstms

    def _construct_new(self, regularization) -> Model:
        raise NotImplementedError("_construct_new was not overridden for the derived class of MLBDeepModel.")

    def train(self, matrices: SequenceMatrices, return_best: Union[bool, str] = False, monitor_loss: str = None,
              early_stopping: int = None, *args, **kwargs):
        if isinstance(return_best, bool):
            if return_best:
                return_best = "models/best.h5"
            else:
                return_best = None

        inputs, outputs = matrices.split_input_output()
        inputs = inputs.get_key_matrices()
        outputs = outputs.get_key_matrices()

        # create the sample weights
        is_temporal = self._sample_weight_type() == "temporal"
        sample_weight = {
            layer.name: matrices.get_sample_weights_for(self._layer_name_to_key(layer.name), is_temporal)
            for layer in self.model.output_layers}

        if monitor_loss is None:
            monitor_loss = "val_loss"

        validation_data = None
        if 'validation_data' in kwargs:
            validation_data = kwargs.pop('validation_data')
            val_input, val_output = validation_data.split_input_output()
            validation_data = (val_input.get_key_matrices(),
                               self.__filter_unused_keys(val_output.get_key_matrices()),
                               {layer.name: validation_data.get_sample_weights_for(self._layer_name_to_key(layer.name),
                                                                                   is_temporal) for layer in
                                self.model.output_layers})

        callbacks = [] if return_best is None else [
            self.model_checkpoint_class(filepath=return_best, verbose=1, save_best_only=True, monitor=monitor_loss)]
        if early_stopping is not None:
            callbacks.append(EarlyStopping(monitor=monitor_loss, patience=early_stopping))
        kwargs["batch_size"] = kwargs.pop("batch_size", 32) * self.batch_mult
        self.model.fit(inputs, self.__filter_unused_keys(outputs),
                       validation_data=validation_data, sample_weight=sample_weight, callbacks=callbacks, *args,
                       **kwargs)

    def predict(self, matrices: SequenceMatrices, intermediate_layer: str = None, *args, **kwargs):
        # make sure the layer exists
        if intermediate_layer is not None and not self.contains_layer(intermediate_layer):
            return {}

        inputs, _ = matrices.split_input_output()
        inputs = inputs.get_key_matrices()

        if intermediate_layer is None:
            model = self.model
        else:
            model = KerasModel(inputs=[layer.input for layer in self.model.input_layers],
                               outputs=self.model.get_layer(intermediate_layer).output)

        # run the prediction
        results = model.predict(inputs, *args, **kwargs)
        if intermediate_layer is not None:
            return {intermediate_layer: results}

        if type(results) is not list:
            results = [results]

        # if there are multiple layers with the same name except a .# at the end, only keep the one with the smallest #
        layer_results = {self.model.output_layers[i].name: results[i] for i in range(len(results))}
        final_results = {}
        for layer_name, result in sorted(layer_results.items()):
            this_key = self._layer_name_to_key(layer_name)
            if this_key not in final_results:
                final_results[this_key] = result
        return final_results

    @staticmethod
    def _layer_name_to_key(layer_name):
        return layer_name if "." not in layer_name else layer_name[:layer_name.rfind(".")]

    def summary(self, *args, **kwargs):
        self.model.summary(*args, **kwargs)

    def normalized_input_layers(self):
        all_names = [layer.name for layer in self.model.layers]
        names = [layer.name for layer in self.model.input_layers if layer.name in all_names]
        modified_names = [
            "norm_" + name if "norm_" + name in all_names else name
            for name in names]
        return [self.model.get_layer(name) for name in modified_names]

    def output_names(self):
        layer_results = {layer.name: 1 for layer in self.model.output_layers}
        final_results = {}
        for layer_name in sorted(layer_results.keys()):
            this_key = self._layer_name_to_key(layer_name)
            if this_key not in final_results:
                final_results[this_key] = 1
        return list(final_results.keys())

    def _loss_function(self):
        return "mse"

    def _loss_weights(self, aux_weight: float = 0.2):
        main_outputs = self.output_names()
        dictionary = {}
        for layer in self.model.output_layers:
            if layer.name in main_outputs:
                dictionary[layer.name] = 1
            else:
                dictionary[layer.name] = aux_weight
        return dictionary

    def _sample_weight_type(self):
        # find the maximum number of dimensions for an output layer. if it's 3, use sample_weight_mode=temporal
        max_out_dims = 2
        for layer in self.model.output_layers:
            max_out_dims = max(max_out_dims, len(layer.output.get_shape()))

        return 'temporal' if max_out_dims == 3 else None

    def compile(self):
        self.model.compile(loss=self._loss_function(), loss_weights=self._loss_weights(),
                           sample_weight_mode=self._sample_weight_type(), optimizer=Adam(lr=3e-4))

    def final_activation(self):
        return "linear"

    def _get_klp_variables(self):
        klp_list = []
        for layer in self.model.layers:
            try:
                klp_list.append(
                    tf.get_default_graph().get_operation_by_name(layer.name + '/keras_learning_phase').outputs[0])
            except KeyError:
                # this doesn't exist. no big deal
                pass
        return klp_list

    def input_gradients(self, sample_data: Dict[str, np.ndarray], feature_index: int, timestep_index: int,
                        output_name: str):
        # get the inputs
        input_tensors = [layer.output for layer in self.model.input_layers if self.contains_layer(layer.name)]
        normalized_input_tensors = [layer.output for layer in self.normalized_input_layers()]
        batch_norm_gammas = {
            layer.name.replace("norm_", ""): layer.get_weights()[0] if isinstance(layer,
                                                                                  BatchNormalization) or isinstance(
                layer, NewBatchNormalization) else None for layer in self.normalized_input_layers()}
        input_names = [layer.name for layer in self.model.input_layers if self.contains_layer(layer.name)]

        # get the learning phase variable so we can set it when we need to
        klp_list = self._get_klp_variables()
        klp_val_list = [tf.constant(False, dtype=bool) for _ in klp_list]

        true_to_normalized = {
            in_layer_name: K.function([self.model.get_layer(in_layer_name).input] + klp_list,
                                      [normalized_input_tensors[i]] + klp_val_list)
            for i, in_layer_name in enumerate(input_names)
        }

        # map all the inputs to their normalized values
        norm_data = {
            key: true_to_normalized[key]([sample_data[key], False])[0] for key in true_to_normalized
        }

        # find the output layer we are trying to predict for
        output_layer = self.model.get_layer(output_name)
        output_tensor = output_layer.output

        # find the gradient tensors for this output (this may include nulls, if the graph is disconnected)
        gradients = K.gradients(K.sum(output_tensor[:, timestep_index:timestep_index + 1, feature_index]),
                                normalized_input_tensors)

        # create a function to map some arbitrary input tensors to the gradients
        zipped = [x for x in zip(normalized_input_tensors, gradients, input_names) if x[1] is not None]
        evaluated_gradients = K.function([item[0] for item in zipped] + klp_list + input_tensors,
                                         [item[1] for item in zipped] + klp_val_list + input_tensors)

        # call the function on the given inputs
        sample_inputs = [norm_data[name] for input_tensor, gradient, name in zipped] + ([False] * len(klp_list)) + [
            sample_data[name] for name in input_names]
        function_output = evaluated_gradients(sample_inputs)

        # re-add the null tensors
        all_results = {entry[2]: function_output[i] for i, entry in enumerate(zipped)}

        # normalize everything by the gammas (if they are present)
        for key, item in all_results.items():
            if item is not None and batch_norm_gammas[key] is not None:
                all_results[key] /= batch_norm_gammas[key]

        # now we have the results!
        return all_results

    def feature_gradients(self, sample_data: SequenceMatrices, layer_name: str, output_layer: str):
        # get the learning phase variable so we can set it when we need to
        klp_list = self._get_klp_variables()
        klp_val_list = [tf.constant(False, dtype=bool) for _ in klp_list]

        input_tensors = [layer.output for layer in self.model.input_layers]
        layer_tensor = self.model.get_layer(layer_name).output

        intermediate_data = self.predict(sample_data, layer_name)[layer_name]

        output_layer = self.model.get_layer(output_layer)

        # find the gradient tensor for this output
        sample_weights = np.expand_dims(sample_data.get_default_sample_weights(), axis=-1)
        sample_weights_tensor = tf.convert_to_tensor(sample_weights, dtype=output_layer.output.dtype,
                                                     name="in_sample_weights")
        gradient = sum(
            [K.square(K.gradients(K.sum(output_layer.output[:, :, i:i + 1] * sample_weights_tensor), [layer_tensor])[0])
             for i in range(K.int_shape(output_layer.output)[-1])])
        gradient = K.sum(K.sum(gradient, axis=1), axis=0)

        # create a function to map some arbitrary input tensors to the gradients
        evaluated_gradients = K.function([layer_tensor] + klp_list + input_tensors,
                                         [gradient] + klp_val_list + input_tensors)

        # call the function on the given inputs
        sample_inputs = [intermediate_data] + ([False] * len(klp_val_list)) + [
            sample_data.get_key_matrices()[layer.name] for layer
            in self.model.input_layers]
        function_output = evaluated_gradients(sample_inputs)[0]

        return function_output / la.norm(function_output)

    def contains_layer(self, name):
        for layer in self.model.layers:
            if layer.name == name:
                return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        K.clear_session()
