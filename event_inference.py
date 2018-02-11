from pymlb.data import GameLogs
import numpy as np
import random as rnd
import pickle
from os.path import isfile

connector = GameLogs(token="[hidden]", cache_directory="data_cache")


def create_matrix_labels(events, label: str, keep_label: bool = False):
    # get the columns
    keys = None
    for key in events.keys():
        keys = [field for field in sorted(events[key].keys())]
        break

    matrix = np.array(
        [[value if keep_label or field != label else "" for field, value in sorted(events[key].items())]
         for key in
         sorted(events)],
        dtype=str)
    labels = np.array([events[key][label] for key in sorted(events)], dtype=str)

    return matrix, labels, keys


def load(seasons, training: bool, predicting: bool, ratio: float = 1):
    events = {}
    for season in seasons:
        these_events = connector.list_events(season,
                                             ["batter_finisher_hand", "event_type", "fielded_by", "batted_ball_type",
                                              "bunt_flag", "sh_flag", "sf_flag", "hit_location", "1st_error_player",
                                              "1st_error_type", "play_on_batter", "runner_on_1st_dest",
                                              "runner_on_2nd_dest", "runner_on_3rd_dest", "fielder_with_first_putout",
                                              "fielder_with_first_assist"],
                                             search={
                                                 "batted_ball_type": ("!=" if not predicting else "="),
                                                 "ball_in_play": "1",
                                             })
        if ratio == 1:
            events.update(these_events)
        else:
            events.update({key: value for key, value in these_events.items() if rnd.random() < ratio})

    unique_events = []
    for key, event in sorted(events.items()):
        event = {key: str(value) for key, value in event.items()}

        unique_events.append(event["unique_event"])
        event.pop("unique_event")

        if event["fielded_by"] != "0":
            event["batter_finisher_hand"] += event["fielded_by"]
        else:
            event["batter_finisher_hand"] = ""
        event.pop("fielded_by")

        event["1st_error_player"] = event["1st_error_player"] + event["1st_error_type"]
        event.pop("1st_error_type")

        for field in ["fielder_with_first_putout", "fielder_with_first_assist", "runner_on_1st_dest",
                      "runner_on_2nd_dest", "runner_on_3rd_dest"]:
            if event[field] == "0":
                event[field] = ""

        any_types_added = False

        for field in ["runner_on_1st_dest", "runner_on_2nd_dest", "runner_on_3rd_dest"]:
            if event[field] != "":
                event[field] = event["event_type"] + "_" + event[field]
                any_types_added = True
        if any_types_added:
            event["event_type"] = ""

        events[key] = event

    return create_matrix_labels(events, label="batted_ball_type", keep_label=training) + (unique_events,)


def train_nb(matrix, labels, smoothing: float = 0.1):
    feature_dicts = []
    feature_matrices = []
    label_list = list(sorted(set(labels)))
    label_dict = {value: index for index, value in enumerate(label_list)}
    for feature_index in range(matrix.shape[1]):
        feature_list = list(sorted(filter(lambda x: len(x) > 0, set(matrix[:, feature_index]))))
        feature_dict = {value: index for index, value in enumerate(feature_list)}
        this_feature_matrix = np.ones((len(feature_list), len(label_list))) * smoothing

        # fill the feature matrix
        for value, label in zip(matrix[:, feature_index], labels):
            if value in feature_dict:
                this_feature_matrix[feature_dict[value], label_dict[label]] += 1

        # normalize the columns
        this_feature_matrix /= np.sum(this_feature_matrix, axis=0, keepdims=True)

        # save it
        feature_dicts.append(feature_dict)
        feature_matrices.append(this_feature_matrix)

    # compute the priors
    priors = np.zeros((len(label_list),))
    for label in labels:
        priors[label_dict[label]] += 1
    priors /= np.sum(priors)

    return label_list, feature_dicts, feature_matrices, priors


def predict_nb(model, matrix):
    label_list, feature_dicts, feature_matrices, priors = model

    label_distributions = []
    labels = []
    for row_index in range(matrix.shape[0]):
        probabilities = np.copy(priors)
        for column, feature_index in zip(columns, range(matrix.shape[1])):
            # make sure we have the feature value
            feature_value = matrix[row_index, feature_index]
            if feature_value not in feature_dicts[feature_index]:
                continue

            # update the probabilities
            probabilities *= feature_matrices[feature_index][feature_dicts[feature_index][feature_value], :]
            probabilities /= np.sum(probabilities)
        label_distributions.append(dict(zip(label_list, probabilities)))
        labels.append(label_list[np.argmax(probabilities)])

    return labels, label_distributions


def evaluate(predictions, ground_truths):
    label_list = list(sorted(set(ground_truths) | set(predictions)))
    label_dict = {value: index for index, value in enumerate(label_list)}

    confusion_matrix = np.zeros((len(label_list), len(label_list)), dtype=int)
    for p, g in zip(predictions, ground_truths):
        confusion_matrix[label_dict[g], label_dict[p]] += 1

    return confusion_matrix


model_type = "nb"
file_name = "models/batted_ball_type_" + model_type + ".pkl"
train_function = globals()["train_" + model_type]
predict_function = globals()["predict_" + model_type]

rnd.seed(1337)
if isfile(file_name):
    with open(file_name, "rb") as f:
        trained_model = pickle.load(f)
    print("Loaded Trained Model")
else:
    training_matrix, training_labels, _, _ = load(list(range(2003, 2014)), training=model_type == "nb",
                                                  predicting=False, ratio=1)
    print("Loaded Training")
    trained_model = train_function(training_matrix, training_labels)
    print("Trained")

    with open(file_name, "wb") as f:
        pickle.dump(trained_model, f)

    # evaluate the accuracy
    val_matrix, val_labels, columns, _ = load([2014], training=False, predicting=False)
    print("Loaded Validation")
    predictions, _ = predict_function(trained_model, val_matrix)
    print("Predicted")
    evaluation = evaluate(predictions, val_labels)
    print("Accuracy:")
    print(np.trace(evaluation) / np.sum(evaluation))
    print("Confusion Matrix:")
    print(evaluation)

# create a mapping for unique event => probability for each outcome
test_matrix, _, columns, unique_events = load(list(range(1957, 2018)), training=False, predicting=True)
print("Loaded Testing")
predictions, prediction_distributions = predict_function(trained_model, test_matrix)
print("Generated Predictions")

csv = []  # [["event", "F", "G", "L", "P", "prediction"] + columns]
for event, distribution, prediction, matrix_row in zip(unique_events, prediction_distributions, predictions,
                                                       range(test_matrix.shape[0])):
    # csv.append([event, distribution["F"], distribution["G"], distribution["L"], distribution["P"],
    #             prediction] + test_matrix[matrix_row, :].tolist())
    csv.append([event, distribution["F"], distribution["G"], distribution["L"], distribution["P"]])

with open("models/missing_batted_ball_types.csv", "w") as f:
    for row in csv:
        print(",".join(map(str, row)), file=f)
