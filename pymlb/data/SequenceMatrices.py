from typing import Dict, Optional
import numpy as np
import pickle


class SequenceMatrices:
    def __init__(self, key_matrices: Dict[str, np.ndarray] = None, file_name: str = None, bias: np.ndarray = None):
        self.key_matrices = key_matrices

        self.train_test = None
        if "traintest" in self.key_matrices:
            self.train_test = np.array(self.key_matrices.pop("traintest")[:, 0])

        self.sample_weights = {}
        default_weights = bias if bias is not None else (
            self.get_default_sample_weights() if "in_bias" in self.key_matrices else None)
        for key in self.key_matrices:
            if not key.startswith("out_"):
                continue

            if "sample_weights_" + key in self.key_matrices:
                self.sample_weights[key] = np.array(self.key_matrices["sample_weights_" + key])[:, :, 0]

                if default_weights is not None:
                    # normalize the sample weights so the mean is 1
                    self.sample_weights[key] *= default_weights
                    self.sample_weights[key] *= np.sum(default_weights) / np.sum(self.sample_weights[key])
            else:
                self.sample_weights[key] = self.get_default_sample_weights()

        if file_name is not None:
            self.key_matrices = pickle.load(open(file_name, "rb"))

    def save(self, file_name: str):
        pickle.dump(self.key_matrices, open(file_name, "wb"))

    def import_matrices(self, matrices: 'SequenceMatrices'):
        def update(self_matrices: np.ndarray, other_matrices: np.ndarray):
            # reshape them if needed
            current_max_length = self_matrices.shape[1]
            new_max_length = other_matrices.shape[1]
            min_length_one = self_matrices if current_max_length < new_max_length else other_matrices
            max_length_one = other_matrices if current_max_length < new_max_length else self_matrices
            if current_max_length != new_max_length:
                if self_matrices.ndim == 3:
                    temp = np.zeros((min_length_one.shape[0], max_length_one.shape[1], min_length_one.shape[2]))
                    temp[:, :min_length_one.shape[1], :] = min_length_one
                else:
                    temp = np.zeros((min_length_one.shape[0], max_length_one.shape[1]))
                    temp[:, :min_length_one.shape[1]] = min_length_one
                min_length_one = temp

            return np.vstack([min_length_one, max_length_one])

        for key in matrices.key_matrices.keys():
            if key not in self.key_matrices:
                self.key_matrices[key] = matrices.key_matrices[key]
            else:
                self.key_matrices[key] = update(self.key_matrices[key], matrices.key_matrices[key])
        for key in matrices.sample_weights.keys():
            if key not in self.sample_weights:
                self.sample_weights[key] = matrices.sample_weights[key]
            else:
                self.sample_weights[key] = update(self.sample_weights[key], matrices.sample_weights[key])

        if matrices.train_test is None or self.train_test is None:
            self.train_test = self.train_test if self.train_test is not None else matrices.train_test
        else:
            self.train_test = np.concatenate([self.train_test, matrices.train_test])

    def get_key_matrices(self):
        # get the arrays except for the sample_weights features which don't go into the model
        copy = dict(self.key_matrices.items())
        for key in list(copy.keys()):
            if key.startswith("out_") and "sample_weights_" + key in copy:
                copy.pop("sample_weights_" + key)
        return copy

    def remove_key(self, key_name: str):
        if key_name in self.key_matrices:
            self.key_matrices.pop(key_name)

    def get_sample_weights(self):
        return self.sample_weights

    def get_default_sample_weights(self):
        return np.squeeze(self.key_matrices["in_bias"], axis=-1)

    def get_sample_weights_for(self, output_key: str, is_temporal: bool):
        if is_temporal == (self.sample_weights[output_key].ndim == 2):
            return self.sample_weights[output_key]
        elif not is_temporal:
            return self.sample_weights[output_key][:, 0]
        else:
            raise ValueError("This output is not three-dimensional, so you must have is_temporal=False.")

    def get_max_sequence_length(self):
        if len(list(self.key_matrices.keys())) == 0:
            return 0
        else:
            for key in self.key_matrices.keys():
                if self.key_matrices[key].ndim == 3:
                    return self.key_matrices[key].shape[1]
            return None

    def get_sequence_count(self):
        if len(list(self.key_matrices.keys())) == 0:
            return 0
        else:
            for key in self.key_matrices.keys():
                return self.key_matrices[key].shape[0]

    def get_entry_count(self):
        if "in_bias" in self.key_matrices:
            return np.sum(self.key_matrices["in_bias"])
        else:
            return None

    def __set_max_sequence_length(self, new_length):
        for key in self.key_matrices.keys():
            if self.key_matrices[key].ndim == 3:
                new_matrix = np.zeros((self.key_matrices[key].shape[0], new_length, self.key_matrices[key].shape[2]))
                new_matrix[:, :self.key_matrices[key].shape[1], :] = self.key_matrices[key]
                self.key_matrices[key] = new_matrix
        for key in self.sample_weights.keys():
            if key in self.sample_weights and self.sample_weights[key].ndim == 2:
                new_matrix = np.zeros((self.sample_weights[key].shape[0], new_length))
                new_matrix[:, :self.sample_weights[key].shape[1]] = self.sample_weights[key]
                self.sample_weights[key] = new_matrix

    def get_keys(self):
        return list(sorted(self.key_matrices.keys()))

    def get_key_counts(self):
        return {key: tuple([None] * (len(self.key_matrices[key].shape) - 2) + [self.key_matrices[key].shape[-1]]) for
                key in self.key_matrices.keys() if not key.startswith("sample_weights_out_")}

    def split_train_test(self):
        # if there are no keys, then just return empty sequences
        if len(self.key_matrices) == 0:
            return SequenceMatrices({}), SequenceMatrices({})

        # if there wasn't a train/test split given, treat the whole thing as train data
        if self.train_test is None:
            return self, SequenceMatrices({})

        # there is a split given.  split it up
        test_indices = np.nonzero(self.train_test)
        train_indices = np.nonzero(self.train_test == 0)

        # TODO: what if in_bias isn't there?
        train_matrices = {"traintest": np.zeros(self.key_matrices["in_bias"].shape[:-1])}
        test_matrices = {"traintest": np.ones(self.key_matrices["in_bias"].shape[:-1])}

        for key, matrices in self.key_matrices.items():
            if matrices.ndim == 3:
                train_matrices[key] = matrices[train_indices, :, :][0, :, :, :]
                test_matrices[key] = matrices[test_indices, :, :][0, :, :, :]
            elif matrices.ndim == 2:
                train_matrices[key] = matrices[train_indices, :][0, :, :]
                test_matrices[key] = matrices[test_indices, :][0, :, :]

        train_matrices = SequenceMatrices(train_matrices)
        return train_matrices, SequenceMatrices(test_matrices)

    def split_input_output(self):
        chain_inputs = {}
        chain_outputs = {}
        for key in self.key_matrices.keys():
            if key.startswith("in_"):
                chain_inputs[key] = self.key_matrices[key]
            else:
                # output vector or sample weights
                chain_outputs[key] = self.key_matrices[key]

        for key in list(chain_outputs.keys()):
            if key.startswith(
                    "out_") and "sample_weights_" + key not in chain_outputs and "in_bias" in self.key_matrices:
                chain_outputs["sample_weights_" + key] = self.key_matrices["in_bias"]

        output_bias = np.squeeze(self.key_matrices["in_bias"], axis=-1) if "in_bias" in self.key_matrices else None
        return SequenceMatrices(chain_inputs), SequenceMatrices(chain_outputs, bias=output_bias)

    def split_fold(self, fold_count):
        sets = [{} for _ in range(fold_count)]
        for key in self.key_matrices.keys():
            for i in range(len(sets)):
                sets[i][key] = self.key_matrices[key][i::len(sets), :, :]
        return tuple(SequenceMatrices(set_item) for set_item in sets)

    def shuffle(self):
        ordering = np.arange(self.get_sequence_count())
        np.random.shuffle(ordering)
        for key in self.key_matrices:
            self.key_matrices[key] = self.key_matrices[key][ordering, :, :]

    def flatten_all(self, recursive_timestep_distance: int = 1, interaction_order: int = 1,
                    remove_zero_samples: bool = True, missing_value_replacement: Optional[float] = 0):
        def add_interactions(matrix: np.ndarray):
            # (a, ..., z, N) => (a, ..., z, N(N-1)/2)
            for order in range(interaction_order - 1):
                columns = [matrix]
                for i in range(matrix.shape[-1]):
                    if matrix.ndim == 2:
                        columns.append(matrix[:, i:i + 1] * matrix[:, i:])
                    elif matrix.ndim == 3:
                        columns.append(matrix[:, :, i:i + 1] * matrix[:, :, i:])
                matrix = np.concatenate(columns, axis=-1)
            return matrix

        # for any time-specific input (in_out_*), create a version that flattens out the last N timesteps into one
        modified_matrices = {}
        sample_weights = {}
        already_added_interactions = set()
        for key, matrix in self.key_matrices.items():
            if key == "in_bias":
                continue

            if key.startswith("in_"):
                # remove the '1' flag at the beginning of all inputs
                if matrix.ndim == 3:
                    matrix = matrix[:, :, 1:]
                elif matrix.ndim == 2:
                    matrix = matrix[:, 1:]

            if matrix.ndim == 3 and key.startswith("in_out_"):
                # add interactions before doing the roll
                if interaction_order > 1:
                    matrix = add_interactions(matrix)
                    already_added_interactions.add(key)

                # perform time-step rolling in these cases
                rolls = [np.roll(matrix, shift=roll, axis=1) for roll in range(recursive_timestep_distance)]
                for i in range(len(rolls)):
                    rolls[i][:, :i, :] = missing_value_replacement
                matrix = np.concatenate(rolls, axis=-1)

            modified_matrices[key] = matrix

            if key.startswith("out_"):
                sample_weights[key] = self.get_sample_weights_for(key, is_temporal=True)

        # flatten everything to 2D if it's not already 2D
        for key, matrix in modified_matrices.items():
            if matrix.ndim == 3:
                modified_matrices[key] = np.reshape(matrix, (-1, matrix.shape[-1]))
        for key, matrix in sample_weights.items():
            if matrix.ndim == 2:
                sample_weights[key] = matrix.flatten()

        if remove_zero_samples:
            # filter out any rows where sample_weight == 0 for all keys
            nonzero_index_list = [set(np.argwhere(weight_matrix > 0).flatten().tolist()) for weight_matrix in
                                  sample_weights.values()]
            nonzero_indices = set()
            for index_list in nonzero_index_list:
                nonzero_indices |= index_list
            nonzero_indices = list(nonzero_indices)
            for key, matrix in modified_matrices.items():
                modified_matrices[key] = matrix[nonzero_indices, :]
            for key, matrix in sample_weights.items():
                sample_weights[key] = matrix[nonzero_indices]

        # add interactions, if needed
        if interaction_order > 1:
            for key, matrix in modified_matrices.items():
                if not key.startswith("in_") or key in already_added_interactions:
                    continue

                modified_matrices[key] = add_interactions(matrix)

        return modified_matrices, sample_weights
