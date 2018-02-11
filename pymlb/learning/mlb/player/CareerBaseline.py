from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import GRU, Masking, Input
from pymlb.learning import MagicVectorDecoderModel
from keras.layers import concatenate


class CareerBaseline(MagicVectorDecoderModel):
    def __init__(self, recurrent_layers: int = 1, *args, **kwargs):
        self.recurrent_layers = recurrent_layers
        super().__init__(*args, **kwargs)

    def _construct_new(self, regularization):
        keys = sorted(self.key_counts.keys())

        # create the input layers
        input_layers = {
            key: Input(name=key, shape=self.key_counts[key]) for key in keys if key.startswith("in_")
        }

        # mask each one
        masked_inputs = {
            key: input_layers[key] if len(input_layers[key].get_shape()) < 3 else self._no_flag(
                Masking(name="masked_" + key)(input_layers[key])) for key in input_layers.keys()
        }

        # normalize each one
        normalized_inputs = {
            key: BatchNormalization(name="norm_" + key)(masked_inputs[key]) for key in
            masked_inputs.keys() if key != "in_bias"
        }

        selected_lstms = [normalized_inputs[key] for key in sorted(normalized_inputs.keys())]
        if len(selected_lstms) > 1:
            selected_lstms = [concatenate(selected_lstms)]

        for i in range(self.recurrent_layers):
            selected_lstms = self._add_lstm_layers(selected_lstms, kind=GRU)

        outs = self._magic_vector_decoders(selected_lstms[0])

        # build the aggregate model
        inputs_list = [input_layers[key] for key in keys if key.startswith("in_")]
        aggregate_model = Model(inputs=inputs_list, outputs=outs)

        return aggregate_model

    def visualization_settings(self):
        raise NotImplementedError
