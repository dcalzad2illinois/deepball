from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, GRU, TimeDistributed, Masking, Input, Activation, Embedding, Flatten
from pymlb.learning import MagicVectorDecoderModel
from pymlb.learning.layers import NewBatchNormalization
from keras.regularizers import l1
from keras.layers.merge import add, concatenate


class CareerModel(MagicVectorDecoderModel):
    def __init__(self, time_dependencies: bool = True, *args, **kwargs):
        kwargs.pop("reuse_structure", "")
        self.time_dependencies = time_dependencies
        super().__init__(reuse_structure=False, *args, **kwargs)

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

        # add noise to the input stats
        # masked_inputs["in_out_stats_extended"] = GaussianNoise(stddev=0.5)(masked_inputs["in_out_stats_extended"])

        # normalize each one
        normalized_inputs = {
            key: BatchNormalization(name="norm_" + key)(masked_inputs[key]) for key in
            masked_inputs.keys() if key != "in_bias"
        }

        normalized_inputs["in_out_stats_extended"] = NewBatchNormalization(name="norm_in_out_stats_extended",
                                                                           center=False, scale=False)(
            masked_inputs["in_out_stats_extended"])

        # add noise to specific ones
        for key in ["in_out_league_offense", "in_previous_year_offense"]:
            if key in normalized_inputs:
                normalized_inputs[key] = NewBatchNormalization(name="norm_" + key, center=False, scale=False)(
                    masked_inputs[key])
                normalized_inputs[key] = GaussianNoise(0.1, name="rand_" + key)(normalized_inputs[key])

        # dense-ify each one
        dense = {
            key:
                Activation("tanh")(TimeDistributed(
                    Dense(int(normalized_inputs[key].get_shape()[-1] * 2)),
                    name='dense_' + key)(normalized_inputs[key])) for key in normalized_inputs
        }
        for key in ["in_out_league_offense", "in_previous_year_offense", "in_out_season_embedding",
                    "in_out_pitch_zones"]:
            if key in normalized_inputs:
                dense[key] = Activation("tanh")(TimeDistributed(Dense(int(normalized_inputs[key].get_shape()[-1])),
                                                                name='dense_' + key)(normalized_inputs[key]))

        # create the other dense layers
        divisions = {
            key: Activation("tanh")(TimeDistributed(
                Dense(int(normalized_inputs[key].get_shape()[-1]) * 3, kernel_regularizer=l1(0.0001),
                               bias_regularizer=l1(0.0001), activation="linear"), name='div_' + key)(
                normalized_inputs[key]))
            for key in
            ["in_out_stats_extended", "in_out_plate_discipline", "in_out_individual_defense",
             "in_out_smoothed_season", "in_out_running"] if key in dense
        }

        # add dropout to all the dense/division layers without the previous league stats
        divisions = {
            key: self._add_dropout_layers([divisions[key]])[0]
            for key in divisions.keys()
        }
        dense = {
            key: self._add_dropout_layers([dense[key]])[0]
            for key in dense.keys()
        }

        divisions = {
            key: self._add_dense_layers([divisions[key]], activation="linear", dimension_multiplier=0.334)[0] for key in
            sorted(divisions.keys())
        }

        # continuous inputs and their associated highway layers
        input_vectors = [dense[key] for key in
                         ["in_bio", "in_park_factors", "in_team_defense", "in_previous_year_offense"] if
                         key in dense]

        stats_park = concatenate(
            [divisions["in_out_stats_extended"], dense["in_out_league_offense"], dense["in_out_park_factors"]] + (
                [dense["in_out_saber"]] if "in_out_saber" in dense else []))

        # lstm for time-sequential inputs
        selected_lstms = [Activation("tanh")(TimeDistributed(Dense(48))(stats_park))] + [
            (divisions[key] if key in divisions else dense[key])
            for key in ["in_out_plate_discipline", "in_out_individual_defense", "in_out_running"] if key in divisions]
        if "in_out_season_embedding" in dense:
            selected_lstms.append(dense["in_out_season_embedding"])
        if "in_out_fielding_position" in dense:
            selected_lstms.append(dense["in_out_fielding_position"])
        if "in_out_pitch_zones" in dense:
            selected_lstms.append(dense["in_out_pitch_zones"])

        selected_lstms = self._add_lstm_layers(selected_lstms,
                                               kind=GRU) if self.time_dependencies else self._add_dense_layers(
            selected_lstms, activation="tanh")
        input_vectors += selected_lstms

        mul_size = self._magic_vector_size()
        embedding_size = 8

        # if they passed the fielding position input, map that to an embedding and use that instead of the dense
        if "in_fielding_position" in masked_inputs:
            embedding = TimeDistributed(Embedding(11, embedding_size, embeddings_initializer="zero"),
                                        name="embed_in_fielding_position")(
                masked_inputs["in_fielding_position"])
            embedding = TimeDistributed(Flatten(), name="norm_in_fielding_position")(embedding)
            input_vectors.append(embedding)

        # if they passed the fielding position input, map that to an embedding and use that instead of the dense
        if "in_age" in masked_inputs:
            embedding = TimeDistributed(Embedding(65, embedding_size, embeddings_initializer="zero"),
                                        name="embed_in_age")(masked_inputs["in_age"])
            embedding = TimeDistributed(Flatten(), name="norm_in_age")(embedding)
            input_vectors.append(embedding)

        # combine everything into a fixed-length vector and do element-wise sum
        combined = add(self._add_dense_layers(input_vectors, out_dim=mul_size, activation='tanh'))

        # add a dense for this layer
        combined = self._add_dropout_layers([combined])[0]
        combined = self._add_dense_layers([combined], 1, activation="tanh")[0]
        combined = Activation("linear", name="magic_vector")(combined)

        outs = self._magic_vector_decoders(combined)

        # build the aggregate model
        inputs_list = [input_layers[key] for key in keys if key.startswith("in_")]
        aggregate_model = Model(inputs=inputs_list, outputs=outs)

        return aggregate_model
