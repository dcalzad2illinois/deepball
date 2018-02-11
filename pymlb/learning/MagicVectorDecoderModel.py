import keras.backend as K
from keras.models import Model
from keras.layers import TimeDistributed, Activation, MaxoutDense, Lambda
from keras.engine.topology import load_weights_from_hdf5_group_by_name
from pymlb.learning import DeepModel
from pymlb.learning.custom_objectives import percentile_loss, gaussian_loss
from pymlb.learning.custom_activations import gaussian
import h5py
import fnmatch


class MagicVectorDecoderModel(DeepModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __create_mv_decoder(self, name: str, suffix: str, trainable: bool, input, override_activation=None):
        activation = override_activation if override_activation is not None else self.final_activation()

        if len(input.get_shape()) == 3:
            # the input we received has a time dimension
            time_layer = TimeDistributed(MaxoutDense(self.key_counts[name][-1], trainable=trainable, nb_feature=8),
                                         name="mv_" + name + suffix)(input)

            if len(self.key_counts[name]) == 2:
                # the output also has a time dimension
                return Activation(activation, name=name + suffix)(time_layer)
            else:
                # the output does not have a time dimension. take the mean of each timestep
                def timestep_mean(x, mask):
                    mask = K.cast(mask, K.floatx())
                    return K.sum(x * K.expand_dims(mask), axis=1) / K.sum(mask, axis=1, keepdims=True)

                return Lambda(timestep_mean, output_shape=lambda x: (x[0], x[2]), mask=lambda x, mask: None,
                              name=name + suffix)(Activation(activation)(time_layer))
        else:
            # the input does NOT have a time dimension

            if len(self.key_counts[name]) == 2:
                # the output DOES have a time dimension
                raise ValueError("You cannot create a magic vector decoder that takes a time-invariant tensor and "
                                 "returns a time-sensitive tensor.")
            else:
                # neither does the output!
                return Activation(activation, name=name + suffix)(
                    MaxoutDense(self.key_counts[name][-1], trainable=trainable, name="mv_" + name + suffix,
                                nb_feature=8)(input))

    def _magic_vector_decoders(self, magic_vector_tensor, trainable: bool = True):
        # create the mean output layers
        outs = []
        if "out_stats" in self.key_counts:
            outs.append(self.__create_mv_decoder("out_stats", "", trainable, magic_vector_tensor))
        if "out_counts" in self.key_counts:
            outs.append(self.__create_mv_decoder("out_counts", "", trainable, magic_vector_tensor))
        if "out_mean_covariance" in self.key_counts:
            outs.append(self.__create_mv_decoder("out_mean_covariance", "", trainable, magic_vector_tensor,
                                                 override_activation=gaussian))

        for i in self._percentiles():
            if "out_stats" in self.key_counts:
                outs.append(
                    self.__create_mv_decoder("out_stats", ".pct" + str(i), trainable, magic_vector_tensor))
            if "out_counts" in self.key_counts:
                outs.append(
                    self.__create_mv_decoder("out_counts", ".pct" + str(i), trainable, magic_vector_tensor))

        if "out_fielding_position" in self.key_counts:
            outs.append(self.__create_mv_decoder("out_fielding_position", "", trainable, magic_vector_tensor))
        if "out_game_counts" in self.key_counts:
            outs.append(self.__create_mv_decoder("out_game_counts", "", trainable, magic_vector_tensor))

        return outs

    def _percentiles(self):
        return []  # [10, 25, 40, 50, 60, 75, 90]

    def _loss_weights(self, aux_weight: float = 0.1):
        weights = {}
        if "out_stats" in self.key_counts:
            weights.update({"out_stats": 1})
        if "out_counts" in self.key_counts:
            weights["out_counts"] = 1
        if "out_mean_covariance" in self.key_counts:
            weights["out_mean_covariance"] = 0.05
        if "out_fielding_position" in self.key_counts:
            weights["out_fielding_position"] = 1
        if "out_game_counts" in self.key_counts:
            weights["out_game_counts"] = 1

        for i in self._percentiles():
            if "out_stats" in self.key_counts:
                weights["out_stats.pct" + str(i)] = aux_weight
            if "out_counts" in self.key_counts:
                weights["out_counts.pct" + str(i)] = aux_weight

        return weights

    def _loss_function(self):
        losses = {}
        if "out_stats" in self.key_counts:
            losses.update({"out_stats": "mse"})
        if "out_counts" in self.key_counts:
            losses["out_counts"] = "mse"
        if "out_mean_covariance" in self.key_counts:
            losses["out_mean_covariance"] = gaussian_loss
        if "out_fielding_position" in self.key_counts:
            losses["out_fielding_position"] = "mse"
        if "out_game_counts" in self.key_counts:
            losses["out_game_counts"] = "mse"

        for i in self._percentiles():
            if "out_stats" in self.key_counts:
                losses["out_stats.pct" + str(i)] = (
                    lambda x: lambda y_true, y_pred: percentile_loss(y_true, y_pred, 1 - x / 100))(i)
            if "out_counts" in self.key_counts:
                losses["out_counts.pct" + str(i)] = (
                    lambda x: lambda y_true, y_pred: percentile_loss(y_true, y_pred, 1 - x / 100))(i)

        return losses

    def _magic_vector_size(self):
        if "out_fielding_position" in self.key_counts:
            return 85 + 5 * len(self._percentiles())
        elif "out_stats" in self.key_counts:
            return 65 + 5 * len(self._percentiles())
        else:
            return 45 + 5 * len(self._percentiles())

    def _import_mv_weights(self, model: Model, file_name: str):
        f = h5py.File(file_name, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        load_weights_from_hdf5_group_by_name(f,
                                             [layer for layer in model.layers if fnmatch.fnmatch(layer.name, "mv_*")])
