from typing import List, Tuple, Dict, Optional
from pymlb.data import SequenceMatrices
from pymlb.learning import Model
import numpy as np
import pickle


class SequenceTable:
    @staticmethod
    def __player_to_int(player_id: str) -> int:
        id_num = 0
        for c in player_id[:5]:
            id_num = id_num * 27 + (26 if c == "-" else (ord(c) - ord('a')))

        return int(player_id[5:]) * (27 ** 5) + id_num

    def first_id(self):
        for i in self.sequences.keys():
            return i
        return None

    @staticmethod
    def first_entry(sequence):
        for i in sorted(sequence.keys()):
            if i != "*":
                return i
        return None

    def __init__(self, sequences=None, file_name=None, include_previous_outputs: bool = True, in_out_roll: int = 1):
        # convert the passed sequences to a dictionary
        if sequences is None:
            self.sequences = {}
        elif type(sequences) is dict:
            self.sequences = sequences
        else:
            self.sequences = {i: sequences[i] for i in range(len(sequences))}

        # convert each sequence to a dictionary
        for sequence_id, sequence in self.sequences.items():
            if type(sequence) is list:
                self.sequences[sequence_id] = {i: sequence[i] for i in range(len(sequence))}

        self.include_previous_outputs = include_previous_outputs
        self.removed_keys = []
        self.input_clones = []
        self.in_out_roll = in_out_roll

        if file_name is not None:
            self.sequences = pickle.load(open(file_name, "rb"))

    def save(self, file_name: str):
        pickle.dump(self.sequences, open(file_name, "wb"))

    def get_sequences(self):
        return self.sequences

    def __len__(self):
        return len(self.sequences)

    def get_keys(self, include_recursive: bool = True, include_removed: bool = False) -> List[str]:
        if len(self.sequences) == 0:
            return []

        first_id = self.first_id()

        # _get a sample entry dictionary
        sample = self.sequences[first_id][self.first_entry(self.sequences[first_id])]
        if "*" in self.sequences[first_id]:
            sample.update(self.sequences[first_id]["*"])

        if self.include_previous_outputs and include_recursive:
            recursive_keys = ["in_" + key for key in sample.keys() if key.startswith("out_")]
        else:
            recursive_keys = []

        keys = list(sample.keys()) + recursive_keys
        keys.append("in_bias")
        if not include_removed:
            keys = list(set(keys) - set(self.removed_keys))
        return sorted(keys)

    def get_keys_counts(self, include_bias: bool = True) -> Dict[str, tuple]:
        # TODO: There is some coupling between this and SequenceMatrices.__sequence_as_matrices() regarding which keys
        # are created.

        keys = self.get_keys(include_recursive=False, include_removed=True)

        first_id = self.first_id()

        # _get a sample entry dictionary
        sample = self.sequences[first_id][self.first_entry(self.sequences[first_id])]
        if "*" in self.sequences[first_id]:
            sample.update(self.sequences[first_id]["*"])

        # _get the counts of the final dimensions
        counts = {key: len(sample[key]) + (1 if include_bias and key.startswith("in_") else 0) for key in keys if
                  key != "in_bias"}

        # for the ones that are unique to each timestep, expand them out
        for key in list(counts.keys()):
            if "*" in self.sequences[first_id] and key in self.sequences[first_id]["*"]:
                counts[key] = (counts[key],)
            else:
                counts[key] = (None, counts[key])

        if self.include_previous_outputs:
            counts.update(
                {"in_" + key: counts[key][:-1] + ((1 if include_bias else 0) + counts[key][-1],) for key in keys if
                 key.startswith("out_")})

        counts["in_bias"] = (None, 1)

        # add the cloned ones
        for cloned in self.input_clones:
            counts["in_clone_" + cloned[4:]] = counts[cloned][:-1] + (counts[cloned][-1] + 1,)

        # remove any one that they explicitly requested to remove
        for removed in self.removed_keys:
            if removed in counts:
                counts.pop(removed)

        return counts

    # obtains a list of dictionaries of matrices, where each entry corresponds to a sequence. also returns a list of
    # corresponding entry IDs
    def __sequences_as_matrices(self, retrieval_sequence_id: str = None, max_entry_id=None, min_entry_id=None,
                                randomize_timesteps: bool = False) -> Tuple[
        Dict[str, Dict[str, np.ndarray]], Dict[str, List[str]]]:
        max_length = self.get_max_sequence_length(min_entry_id, max_entry_id)

        sequence_dictionaries = {}
        sequence_entries = {}
        keys = self.get_keys(include_recursive=False, include_removed=True)
        for sequence_id, sequence in sorted(self.sequences.items()):
            # TODO: optimize this. we don't need to iterate through all the sequences if we know which one we want
            if retrieval_sequence_id is not None and sequence_id != retrieval_sequence_id:
                continue

            # create the matrices for the entry-specific vectors
            sub_sequence = {key: value for key, value in sequence.items() if key != "*"}

            # order the timesteps randomly or in order, depending on what they want
            ordered_subsequence = sub_sequence.items() if randomize_timesteps else sorted(sub_sequence.items())

            # create a dictionary of matrices for all input vectors
            d = {key: np.matrix([([1] if key.startswith("in_") else []) + list(entry[key])
                                 for entry_id, entry in ordered_subsequence])
                 for key in keys if key != "in_bias" and ("*" not in sequence or key not in sequence["*"])}

            if "*" in sequence:
                # now bring in the sequence-specific vectors
                d.update({key: np.array(vector) for key, vector in sequence["*"].items() if key.startswith("in_")})
                d.update(
                    {key: np.repeat(np.matrix(vector), len(sub_sequence), 0) for key, vector in sequence["*"].items() if
                     key.startswith("out_")})

            # finally, add the bias
            d["in_bias"] = np.ones((len(sub_sequence), 1))

            # if they want, include the output vectors from the previous time step
            if self.include_previous_outputs:
                for key in keys:
                    if not key.startswith("out_") or ("*" in sequence and key in sequence["*"]):
                        continue

                    if key in self.input_clones:
                        d["in_clone_" + key[4:]] = np.ones((d[key].shape[0], d[key].shape[1] + 1))
                        d["in_clone_" + key[4:]][:, 1:] = d[key]
                    if ("in_" + key) not in self.removed_keys:
                        # roll the output matrix
                        d["in_" + key] = np.ones((d[key].shape[0], d[key].shape[1] + 1))
                        d["in_" + key][:, 1:] = np.roll(d[key], self.in_out_roll, axis=0)

                        # zero out the first row so it can't see the output of the last time step in the first time step
                        d["in_" + key][:self.in_out_roll, 1:] = 0

            # remove any removed keys
            for removed_key in self.removed_keys:
                if removed_key in d:
                    d.pop(removed_key)

            # now filter the timesteps
            indices = [i for i, key in enumerate(sorted(sub_sequence.keys())) if
                       (max_entry_id is None or key <= str(max_entry_id)) and (
                           min_entry_id is None or key >= str(min_entry_id))]
            for key in d.keys():
                if d[key].ndim == 2:
                    d[key] = d[key][indices, :]

            # reshape all the arrays to include the sequence dimension and match the length
            for key in d.keys():
                if d[key].ndim == 2:
                    new_matrix = np.zeros((1, max_length, d[key].shape[1]))
                    new_matrix[0, :d[key].shape[0], :] = d[key]
                    d[key] = new_matrix
                elif d[key].ndim == 1:
                    new_matrix = np.zeros((1, d[key].shape[0]))
                    new_matrix[0, :] = d[key]
                    d[key] = new_matrix

            if "traintest" in d:
                d["traintest"] = 1 if np.sum(d["traintest"], axis=-2) > 0 else 0

            sequence_dictionaries[sequence_id] = d
            sequence_entries[sequence_id] = [key for key, value in ordered_subsequence]
        return sequence_dictionaries, sequence_entries

    def get_max_sequence_length(self, min_entry_id=None, max_entry_id=None):
        if len(self.sequences) == 0:
            return 0
        else:
            return max([len([sequence_item for sequence_item in sequence.keys() if
                             (min_entry_id is None or sequence_item >= str(min_entry_id)) and (
                                 max_entry_id is None or sequence_item <= str(max_entry_id))]) for sequence in
                        self.sequences.values()])

    def __get_matrices(self, retrieval_sequence_id: str = None, max_entry_id=None, min_entry_id=None,
                       randomize_timesteps: bool = False) -> Tuple[SequenceMatrices, List[List[Tuple[str, Optional[str]]]]]:
        # _get each sequence as a dictionary of lists
        matrices = {}

        items = []
        matrices_map = self.__sequences_as_matrices(retrieval_sequence_id, max_entry_id, min_entry_id,
                                                    randomize_timesteps=randomize_timesteps)
        max_sequence = self.get_max_sequence_length(min_entry_id, max_entry_id)
        for sequence_id, dictionary in sorted(matrices_map[0].items()):
            # skip sequences with 0 timesteps
            if "traintest" in dictionary and dictionary["in_bias"].shape[1] == 0:
                continue

            for key, matrix in dictionary.items():
                # add this matrix to the list of matrices for this key
                if key not in matrices:
                    matrices[key] = []
                matrices[key].append(matrix)

            items.append([(sequence_id, entry_id) for entry_id in matrices_map[1][sequence_id]] +
                         [(sequence_id, None) for _ in
                          range(max_sequence - len(matrices_map[1][sequence_id]))])

        return SequenceMatrices({key: np.vstack(matrices[key]) for key in matrices.keys()}), items

    def as_matrices(self, *args, **kwargs) -> SequenceMatrices:
        return self.__get_matrices(*args, **kwargs)[0]

    def matrices_index_map(self, *args, **kwargs) -> List[List[Tuple[str, str]]]:
        return self.__get_matrices(*args, **kwargs)[1]

    def remove_key(self, key: str):
        self.removed_keys.append(key)

    def add_input_clone(self, key: str):
        self.input_clones.append(key)

    def filter_entries(self, predicate):
        for sequence_id, sequence in list(self.sequences.items()):
            # remove any entries that fail the predicate
            for entry_id, entry in list(sequence.items()):
                if not predicate(sequence_id, entry_id):
                    sequence.pop(entry_id)

            # remove the sequence if it's empty
            if len(sequence) == 0:
                self.sequences.pop(sequence_id)

    def add_key(self, column_name: str, vector_retriever):
        for sequence_id, sequence in self.sequences.items():
            for entry_id, entry in sequence.items():
                new_item = vector_retriever(sequence_id, entry_id)
                if isinstance(new_item, np.ndarray):
                    new_item = new_item.tolist()
                entry[column_name] = new_item

    def get_entry(self, sequence_id, entry_id=None):
        if entry_id is None:
            return self.sequences[sequence_id]
        elif sequence_id in self.sequences and entry_id in self.sequences[sequence_id]:
            return self.sequences[sequence_id][entry_id]
        else:
            return None

    @staticmethod
    def get_all_matrices(sequences: List['SequenceTable']):
        all_matrices = SequenceMatrices({})
        for sequence in sequences:
            all_matrices.import_matrices(sequence.as_matrices())
        return all_matrices

    @staticmethod
    def _components_first(dictionary):
        result = {}
        for sequence_id, sequence in dictionary.items():
            for entry_id, entry in sequence.items():
                for key, entry_component in entry.items():
                    if key not in result:
                        result[key] = {}
                    if sequence_id not in result[key]:
                        result[key][sequence_id] = {}
                    result[key][sequence_id][entry_id] = entry_component
        return result

    @staticmethod
    def _components_last(dictionary):
        result = {}
        for key, sequences in dictionary.items():
            for sequence_id, sequence in sequences.items():
                if sequence_id not in result:
                    result[sequence_id] = {}
                for entry_id, entry_component in sequence.items():
                    if entry_id not in result[sequence_id]:
                        result[sequence_id][entry_id] = {}
                    result[sequence_id][entry_id][key] = entry_component
        return result

    def get_components_first(self):
        return self._components_first(self.sequences)

    def map(self, model: Model, intermediate_layer: str = None, components_first: bool = True):
        # generate the predictions
        prediction_matrices = model.predict(self.as_matrices(), intermediate_layer=intermediate_layer)
        if len(prediction_matrices) == 0:
            return {}

        # split the predictions into buckets by the number of dimensions
        prediction_buckets = {2: {}, 3: {}}
        for key, tensor in prediction_matrices.items():
            if tensor.ndim > max(prediction_buckets.keys()):
                # this prediction has too many dimensions. flatten the last dimensions into one
                tensor = np.reshape(tensor, tensor.shape[:2] + (-1,))
            elif tensor.ndim < min(prediction_buckets.keys()):
                # this prediction has too few dimensions. expand the last dimensions by repeating (normally this won't
                # be the case) (this also relies on the fact that min(bucket keys) = 2 otherwise it wouldn't know how
                # many dimensions to add)
                tensor = np.repeat(tensor, self.get_max_sequence_length(), axis=-1)

            prediction_buckets[tensor.ndim][key] = tensor

        final_map = {key: {} for key in prediction_matrices.keys()}
        for sequence_index, sequence_id in enumerate(sorted(self.sequences.keys())):
            for key in final_map:
                final_map[key][sequence_id] = {}

            # add in all the two-dimensional predictions
            for key, prediction in prediction_buckets[2].items():
                final_map[key][sequence_id]["*"] = prediction[sequence_index, :]

            # now the three-dimensional predictions
            for entry_index, entry_id in enumerate(
                    sorted(filter(lambda x: x != "*", self.sequences[sequence_id].keys()))):
                for key, prediction in prediction_buckets[3].items():
                    final_map[key][sequence_id][entry_id] = prediction[sequence_index, entry_index, :]

        if not components_first:
            final_map = self._components_last(final_map)

        return final_map

    def boost_sample_weights(self, output_layer: str, previous_model: Model):
        weights_key = "sample_weights_" + output_layer
        if weights_key in self.get_keys(include_recursive=True, include_removed=False):
            previous_predictions = self.map(previous_model, intermediate_layer=output_layer)[output_layer]

            def new_weights(sequence_id, entry_id):
                old_weights = self.sequences[sequence_id][entry_id][weights_key][0]
                residual = np.mean(np.square(
                    np.array(self.sequences[sequence_id][entry_id][output_layer]) - previous_predictions[sequence_id][
                        entry_id]))
                return [old_weights * (old_weights * residual + 1),
                        self.sequences[sequence_id][entry_id][weights_key][1]]

            self.add_key(weights_key, new_weights)
