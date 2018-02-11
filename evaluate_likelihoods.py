from pymlb.data import VectorSlots, SequenceTable, SequenceMatrices
import numpy as np
import numpy.linalg as la

connector = VectorSlots(token="[hidden]", cache_directory="data_cache")
input_slots = 18
output_slots = 7


# https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy/2415343
def weighted_var(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average(np.square(values - np.expand_dims(average, axis=axis)), weights=weights, axis=axis)
    return average, variance


def likelihood(value, mean, var):
    dims = len(value)

    if var.ndim == 1:
        log_det_precision = -np.sum(np.log(var))
    else:
        log_det_precision = np.log(la.det(la.inv(var)))

    return -0.5 * (dims * np.log(2 * np.pi) - log_det_precision + np.sum(np.square(value - mean) / var))


def load(parameters):
    # fetch the results
    chains = SequenceTable(connector.get_vector_values(input_slots, parameters), include_previous_outputs=False)

    matrices = chains.as_matrices()
    matrices.remove_key("traintest")
    matrices.remove_key("in_bias")
    matrices.remove_key("out_counts")

    # extract the deepball counts
    deepball_pas = matrices.get_key_matrices()["in_deepball_counts"][:, :, 1].flatten()
    matrices.remove_key("in_deepball_counts")

    # extract the deepball normal variances
    deepball_variances = matrices.get_key_matrices()["in_deepball_variance"][:, :, 1:6]
    deepball_variances = np.reshape(deepball_variances, (-1, deepball_variances.shape[-1]))
    matrices.remove_key("in_deepball_variance")

    # separate the ground truths, sample weights, and the models
    sample_weights = matrices.get_sample_weights_for("out_ground_truth", is_temporal=True).flatten()

    # first, rescale everything to mean=1
    models = {}
    for key in matrices.get_keys():
        if key[:3] != "in_" and key[:4] != "out_":
            continue
        models[key] = np.reshape(matrices.get_key_matrices()[key], (-1, matrices.get_key_matrices()[key].shape[-1]))
        if key[:3] == "in_":
            models[key] = models[key][:, 1:]

        this_mean = np.expand_dims(np.average(models[key], weights=sample_weights, axis=0), axis=0)
        if key == "in_steamer":
            models[key] /= this_mean

        if key == "in_deepball_normal":
            # now that we have the mean, divide the variances by this squared
            deepball_variances /= np.square(this_mean)

    ground_truths = models.pop("out_ground_truth")

    # get rid of all examples where weight == 0
    indices = np.argwhere(sample_weights > 0).flatten()
    sample_weights = sample_weights[indices]
    deepball_pas = deepball_pas[indices]
    deepball_variances = deepball_variances[indices]
    ground_truths = ground_truths[indices, :]
    for key in models:
        models[key] = models[key][indices, :]

    sequence_entry_map = np.array([item for sequence_list in chains.matrices_index_map() for item in sequence_list])[indices].tolist()
    return chains, sequence_entry_map, ground_truths, sample_weights, deepball_pas, deepball_variances, models


def find_optimal_weights(in_models, ground_truths, sample_weights):
    # models["in_deepball_hamster"] = np.zeros_like(ground_truths)
    w = []
    for stat_index in range(ground_truths.shape[-1]):
        these_stats = np.stack([in_models[key][:, stat_index] for key in sorted(in_models.keys())], axis=-1)
        these_truths = ground_truths[:, stat_index]
        these_stats_weighted = these_stats * np.expand_dims(np.sqrt(sample_weights), axis=1)
        these_truths_weighted = these_truths * np.sqrt(sample_weights)
        weights = la.lstsq(these_stats_weighted, these_truths_weighted)
        w.append(weights)

        # models["in_deepball_hamster"][:, stat_index] = np.dot(these_stats, weights[0])

        # for key in set(in_models.keys()) | set(["in_deepball_hamster"]):
        #     print(key)
        #     print(np.average(np.square(models[key] - ground_truths), weights=sample_weights, axis=0))

    # print(np.average(np.square(models["in_deepball"] - ground_truths), weights=sample_weights, axis=0))
    # print(np.average(np.square(models["in_marcel"] - ground_truths), weights=sample_weights, axis=0))
    # print(np.average(np.square(models["in_deepball_hamster"] - ground_truths), weights=sample_weights, axis=0))

    # turn this into model => weights
    w = np.array([item[0] for item in w])
    return {key: w[:, model_index] for model_index, key in enumerate(sorted(in_models.keys()))}


def get_optimal_weights(models, ground_truths, sample_weights, sample_mask):
    expanded_mask = np.expand_dims(sample_mask, axis=-1)
    return find_optimal_weights({key: models[key] * expanded_mask for key in
                                 ["in_marcel", "in_rf", "in_rnn", "in_deepball"]}, ground_truths * expanded_mask,
                                sample_weights * sample_mask)


# load the data
_, sequence_entry_map, ground_truths, sample_weights, deepball_pas, deepball_variances, models = load(
    {"season": "2015|2017"})

# find the optimal weights
optimal_weights_all = get_optimal_weights(models, ground_truths, sample_weights, np.ones_like(sample_weights))


def add_deepball_plus_combination(models, chains=None, sequence_entry_map=None):
    current = np.zeros_like(models["in_deepball"])
    for sample_index in range(sample_weights.shape[0]):
        # train the weights without this sample
        sample_mask = np.ones_like(sample_weights)
        sample_mask[sample_index] = 0
        weights = get_optimal_weights(models, ground_truths, sample_weights, sample_mask)
        for key in weights.keys():
            current[sample_index] += models[key][sample_index, :] * weights[key]

    if current is not None:
        models["in_deepball_plus"] = current

        # also add it to the chains if necessary
        if chains is not None:
            # convert this to a sequence/entry map
            current_dictionary = {}
            for row_index, sequence_entry in enumerate(sequence_entry_map):
                sequence_id, entry_id = sequence_entry
                if entry_id is None:
                    continue
                if sequence_id not in current_dictionary:
                    current_dictionary[sequence_id] = {}
                current_dictionary[sequence_id][entry_id] = current[row_index, :]

            chains.add_key("in_deepball_plus", lambda sequence_id, entry_id: current_dictionary[sequence_id][entry_id])
            return chains.get_components_first()["in_deepball_plus"]

    return None


# now create a new model for this
add_deepball_plus_combination(models)

# find if we can come up with a variable weighting based on our guess of the player's PAs
w = []
for i in range(5):
    X = np.array([
        [P * (N - G), N - G] for P, N, G in
        zip(deepball_pas, models["in_deepball"][:, i], models["in_deepball_normal"][:, i])
    ])
    y = np.array([
        T - G for T, G in zip(ground_truths[:, i], models["in_deepball_normal"][:, i])
    ])

    X *= np.expand_dims(np.sqrt(sample_weights), axis=1)
    y *= np.sqrt(sample_weights)
    w.append(la.lstsq(X, y)[0])


def add_deepball_plus_lpa(models, chains=None):
    models["in_deepball_plus"] = np.array([
        (w_vector[0] * deepball_pas + w_vector[1]) * models["in_deepball"][:, i] + (
            1 - w_vector[0] * deepball_pas - w_vector[1]) * models["in_deepball_normal"][:, i] for i, w_vector in
        enumerate(w)
    ]).T

    # also add it to the chains if necessary
    if chains is not None:
        def retriever(sequence_id, entry_id):
            entry = chains.get_entry(sequence_id, entry_id)
            this_pa = entry["in_deepball_counts"][0]
            return [
                (w_vector[0] * this_pa + w_vector[1]) * entry["in_deepball"][i] + (
                    1 - w_vector[0] * this_pa - w_vector[1]) * entry["in_deepball_normal"][i] for i, w_vector in
                enumerate(w)
            ]

        chains.add_key("in_deepball_plus", retriever)

        mapping = {}
        for sequence_id, sequence in chains.get_sequences().items():
            mapping[sequence_id] = {}
            for entry_id, entry in sequence.items():
                mapping[sequence_id][entry_id] = entry["in_deepball_plus"]
        return mapping
    else:
        return None


# add_deepball_plus_lpa(models)

# for each system, find the standard deviation of the error
model_empirical_mean = {}
model_empirical_var = {}
model_empirical_mse = {}
for key in models:
    errors = models[key] - ground_truths
    model_empirical_mean[key], model_empirical_var[key] = weighted_var(errors, sample_weights, axis=0)
    model_empirical_mse[key] = np.average(np.square(errors), weights=sample_weights, axis=0)

# find the likelihood of each sample under each system
model_empirical_likelihood = {}
for key in models:
    errors = models[key] - ground_truths
    model_empirical_likelihood[key] = np.array([
        likelihood(row, model_empirical_mean[key], model_empirical_var[key]) for row in errors
    ])

    if key == "in_deepball_normal":
        model_empirical_likelihood[key + "2"] = np.array([
            likelihood(row, model_empirical_mean[key], variance) for row, variance in zip(errors, deepball_variances)
        ])

# find bot the unweighted and weighted total likelihood for each
model_empirical_likelihood_total = {key: np.sum(value) for key, value in model_empirical_likelihood.items()}
model_empirical_likelihood_weighted_total = {key: np.sum(value * sample_weights) for key, value
                                             in model_empirical_likelihood.items()}

# sort the examples by likelihood
large_samples = np.argwhere(sample_weights > 1.5)
likelihoods = model_empirical_likelihood["in_deepball_normal2"][large_samples].flatten()
sorted_likelihoods = np.argsort(likelihoods)
sorted_players = np.array(sequence_entry_map)[large_samples.flatten(), :][sorted_likelihoods]

# now use these weights on the test set
chains, sequence_entry_map, ground_truths, sample_weights, deepball_pas, deepball_variances, models = load({"season": "2015|2017"})
mapping = add_deepball_plus_combination(models, chains, sequence_entry_map)
connector.put_vector_slots(output_slots, "out_stats.plus", mapping, pre_normalized=True)

print("hi")
