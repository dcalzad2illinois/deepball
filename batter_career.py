import argparse
from os.path import isfile, isdir
from os import makedirs
import glob
import sys
import numpy as np
import numpy.linalg as la
import warnings

possible_outputs = ["out_stats", "out_counts", "out_mean_covariance", "out_fielding_position", "magic_vector"]

parser = argparse.ArgumentParser(description="Trains, evaluates, and generates predictions for a batter's career")

action_group = parser.add_mutually_exclusive_group(required=True)
action_group.add_argument("--train", action="store_true", help="Trains a model")
action_group.add_argument("--generate", action="store_true",
                          help="Generates predictions from a model without uploading")
action_group.add_argument("--upload", action="store_true", help="Uploads a model's results to the API")

parser.add_argument("-a", "--aggregating",
                    default=[],
                    metavar="model",
                    nargs="+",
                    help="Which models will be used in aggregation")
parser.add_argument("-ua", "--use-aggregate",
                    action="store_true",
                    help="When using an aggregate model, set this flag.")
parser.add_argument("-t", "--model-type",
                    default="deep-dense",
                    metavar="type",
                    nargs="?",
                    choices=["rf-linear", "simple-rnn", "deep-dense"],
                    help="Which model type will be used.")
parser.add_argument("-f", "--fold",
                    default="4",
                    metavar="fold",
                    nargs="?",
                    choices=list(map(str, range(5))),
                    help="Which cross-validation fold will be used, between 0 and 4 inclusive")
parser.add_argument("--shift",
                    default="1",
                    nargs="?",
                    choices=list(map(str, range(1, 3))),
                    help="The shift offset for the time-sensitive inputs")
parser.add_argument("-m", "--model",
                    nargs="?",
                    help="The target of training or the model to be read/used")
parser.add_argument("-o", "--outputs",
                    default=possible_outputs,
                    nargs="+",
                    metavar="output",
                    choices=possible_outputs,
                    help="Which data types will be trained or generated")
parser.add_argument("-ys", "--start-year",
                    nargs="?",
                    type=int,
                    help="The year at which to start training/evaluating (inclusive)")
parser.add_argument("-ye", "--end-year",
                    nargs="?",
                    type=int,
                    help="The year at which to stop training/evaluating (inclusive)")
parser.add_argument("-p", "--partition",
                    default=None,
                    nargs="?",
                    choices=["all", "train", "validation"],
                    help="Which segment of the data will be used")
parser.add_argument("-us", "--upload-suffix",
                    default=None,
                    nargs="?",
                    help="The suffix to be appended to any uploaded predictions")

args = parser.parse_args()

# do some error checking that the parser couldn't
if args.use_aggregate and args.train:
    warnings.warn("The 'use_aggregate' flag is unnecessary when training. Turning it off.", RuntimeWarning)
    args.use_aggregate = False

if len(args.aggregating) > 0:
    if not args.train:
        raise ValueError("You can only specify 'aggregating' when training an aggregate model.")
    args.aggregating = [item.replace("\\", "/") for parameter in args.aggregating for item in glob.glob(parameter)]

if args.train and len(args.aggregating) == 0 and (args.model_type == "deep-dense" or args.model_type == "simple-rnn"):
    if args.partition and args.partition != "all":
        raise ValueError(
            "For training a neural model, you must use '--partition all' (the validation data will be used "
            "for model selection).")
    elif not args.partition:
        args.partition = "all"

if not args.partition:
    if args.train:
        args.partition = "train"
    else:
        args.partition = "all"

from pymlb.data import SequenceTable, SequenceMatrices, VectorSlots
from pymlb.learning import DeepModel, AggregateModel, RandomForestContinuousModel
from pymlb.learning.mlb.player import CareerAggregateModel, CareerBaseline
from pymlb.learning.utilities import GaussianVisualizer
from pymlb.learning.mlb.player import CareerModel

connector = VectorSlots(token="[hidden]", cache_directory="data_cache")


def eprint(*a, **kwargs):
    print(*a, file=sys.stderr, **kwargs)


def get_vector_set():
    if args.shift == "1":
        return 1
    if args.shift == "2":
        return 7
    raise ValueError


def get_field_names():
    return connector.get_vector_fields(get_vector_set())


def get_default_file(suffix):
    if not isdir("models/bc"):
        makedirs("models/bc")
    if not isdir("models/bc/shift" + args.shift):
        makedirs("models/bc/shift" + args.shift)
    if not isdir("models/bc/shift" + args.shift + "/fold" + args.fold):
        makedirs("models/bc/shift" + args.shift + "/fold" + args.fold)

    return "models/bc/shift" + args.shift + "/fold" + args.fold + "/" + suffix


def get_train_file():
    assert args.train

    if args.model:
        return args.model

    if len(args.aggregating) > 0:
        return get_default_file(args.model_type + "_aggregate.pkl")

    for i in range(100):
        file_name = get_default_file(str(i) + ".h5")
        if not isfile(file_name):
            return file_name
    return None


def get_test_file():
    assert not args.train

    if args.model:
        if not isfile(args.model):
            raise ValueError("The source model '" + args.model + "' does not exist.")

        return args.model

    if len(args.aggregating) > 0:
        return get_default_file(args.model_type + "_aggregate.pkl")

    existing_file = None
    for i in range(100):
        file_name = get_default_file(str(i) + ".h5")
        if not isfile(file_name):
            return existing_file
        existing_file = file_name
    return None


def load(start_year: int = None, end_year: int = None, min_entry_id: int = None, max_entry_id: int = None):
    start_year = start_year or args.start_year or 1958
    end_year = end_year or args.end_year or 2018
    eprint("Years: [" + str(start_year) + "," + str(end_year) + "]")
    eprint("Shift: " + str(args.shift))
    eprint("Fold: " + str(args.fold))

    # load the sequences from the database
    sequences = connector.get_vector_values(get_vector_set(), {
        "in_bio-debut_year": ">=" + str(start_year),
        "season": "<=" + str(end_year),
        "cv_folds": 5,
        "fold": args.fold
    })

    # convert these to chains
    chains = SequenceTable(sequences, in_out_roll=int(args.shift))

    # remove keys we will never use
    chains.remove_key("in_out_stats")
    chains.remove_key("out_plate_discipline")
    chains.remove_key("out_stats_extended")
    chains.remove_key("out_park_factors")
    chains.remove_key("out_season_embedding")
    chains.remove_key("out_league_offense")
    chains.remove_key("out_running")
    chains.remove_key("out_saber")
    chains.remove_key("out_season_embedding")
    chains.remove_key("in_out_counts")
    chains.remove_key("out_pitch_zones")

    def mean_covariance_retriever(sequence_id, entry_id):
        entry = chains.get_entry(sequence_id, entry_id)

        combined = np.concatenate([entry["out_stats"], entry["out_counts"]])
        return np.concatenate([combined, np.zeros((combined.size ** 2,))])

    if "out_mean_covariance" in args.outputs:
        chains.add_key("out_mean_covariance", mean_covariance_retriever)
        chains.remove_key("in_out_mean_covariance")

    for key in possible_outputs:
        if key not in args.outputs:
            chains.remove_key(key)

    # if they gave an only_entry_id, set all sample weights to 0 that aren't of that entry id
    if min_entry_id is not None or max_entry_id is not None:
        if min_entry_id is None:
            min_entry_id = 0
        if max_entry_id is None:
            max_entry_id = 9999
        chains.add_key("sample_weights_out_stats",
                       lambda s, e: [0, 0] if not (min_entry_id <= int(e) <= max_entry_id) else
                       chains.get_entry(s, e)[
                           "sample_weights_out_stats"])

    matrices = chains.as_matrices()
    if args.partition == "train":
        matrices, _ = matrices.split_train_test()
    if args.partition == "validation":
        _, matrices = matrices.split_train_test()
    return chains, matrices


def create_model(file_name: str, matrices: SequenceMatrices):
    single_file_name = None if len(args.aggregating) > 0 or args.use_aggregate else file_name
    if args.model_type == "rf-linear":
        model = RandomForestContinuousModel(key_counts=matrices.get_key_counts(), file_name=single_file_name,
                                            recursive_timestep_distance=4, estimators=60)
    elif args.model_type == "simple-rnn":
        model = CareerBaseline(key_counts=matrices.get_key_counts(), file_name=single_file_name, recurrent_layers=2)

    elif args.model_type == "deep-dense":
        model = CareerModel(file_name=single_file_name, key_counts=matrices.get_key_counts())

    else:
        return None

    if len(args.aggregating) > 0 or args.use_aggregate:
        model = CareerAggregateModel(file_name=file_name, template_single=model)

    model.summary()

    return model


def train():
    _, matrices = load()

    eprint("Training into " + get_train_file())
    encoder = create_model(get_train_file(), matrices)

    # train and save the model
    if isinstance(encoder, DeepModel):
        train_data, test_data = matrices.split_train_test()
        encoder.train(train_data, batch_size=128, verbose=2, epochs=1000, return_best=get_train_file(),
                      validation_data=test_data, early_stopping=50)
    elif isinstance(encoder, RandomForestContinuousModel):
        encoder.train(matrices)
        encoder.save(get_train_file())
    elif isinstance(encoder, AggregateModel):
        for file in args.aggregating:
            eprint("Aggregating from " + file)

        encoder.train(matrices, files=args.aggregating)
        encoder.save(get_train_file())


def upload():
    suffix_primary = "" if args.upload_suffix is None else "." + args.upload_suffix
    suffix_secondary = "" if args.upload_suffix is None else "_" + args.upload_suffix

    chains, matrices = load()
    eprint("Uploading predictions from " + get_test_file())
    encoder = create_model(get_test_file(), matrices)

    print("Generating mapping . . .")

    # create the maps
    maps = chains.map(encoder)
    if "magic_vector" in args.outputs:
        maps.update(chains.map(encoder, intermediate_layer="magic_vector"))

    print("Saving mapping to database . . .")

    # find the mean/covariances
    if "out_mean_covariance" in maps and "out_mean_covariance" in args.outputs:
        mean_covariances = maps["out_mean_covariance"]
        means = {}
        covariances = {}
        variances = {}
        samples = [{"stats": {}, "counts": {}} for _ in range(5)]
        for sequence_id, sequence in mean_covariances.items():
            means[sequence_id] = {}
            covariances[sequence_id] = {}
            variances[sequence_id] = {}
            for item in samples:
                item["stats"][sequence_id] = {}
                item["counts"][sequence_id] = {}
            for entry_id, entry in sequence.items():
                mean = entry[:8]
                means[sequence_id][entry_id] = mean

                # format the covariance matrix correctly
                precision = np.reshape(entry[8:], (8, 8))
                cov = np.linalg.inv(precision)
                covariances[sequence_id][entry_id] = cov.flatten()
                variances[sequence_id][entry_id] = np.diag(cov)

        print("Saving out_stats.normal" + suffix_secondary + " . . .")
        connector.put_vector_slots(get_vector_set(), "out_stats.normal" + suffix_secondary, {
            sequence_id: {entry_id: mean[:5] for entry_id, mean in means[sequence_id].items()} for sequence_id in
            means})
        print("Saving out_counts.normal" + suffix_secondary + " . . .")
        connector.put_vector_slots(get_vector_set(), "out_counts.normal" + suffix_secondary, {
            sequence_id: {entry_id: mean[5:] for entry_id, mean in means[sequence_id].items()} for sequence_id in
            means})
        print("Saving normal_mean" + suffix_secondary + " . . .")
        connector.put_vector_slots(get_vector_set(), "normal_mean" + suffix_secondary, means)
        print("Saving normal_variance" + suffix_secondary + " . . .")
        connector.put_vector_slots(get_vector_set(), "normal_variance" + suffix_secondary, variances)
        print("Saving normal_covariance" + suffix_secondary + " . . .")
        connector.put_vector_slots(get_vector_set(), "normal_covariance" + suffix_secondary, covariances)

        maps.pop("out_mean_covariance")

    # save the all-data mapping and magic vectors into the database
    for layer in args.outputs:
        if layer in maps:
            suffix = suffix_primary if "." not in layer else suffix_secondary
            print("Saving " + layer + suffix + " . . .")
            connector.put_vector_slots(get_vector_set(), layer + suffix, maps[layer])
    print("Mapping saved to database.")


def generate():
    chains, matrices = load()
    eprint("Generating predictions from " + get_test_file())
    encoder = create_model(get_test_file(), matrices)
    fields = get_field_names()

    print("Generating mapping . . .")

    # create the maps
    maps = chains.map(encoder)
    if "magic_vector" in args.outputs:
        maps.update(chains.map(encoder, intermediate_layer="magic_vector"))

    while True:
        batter_id = input(
            "Enter the ID of the batter you would like to obtain predictions for: ").lower().strip()
        if batter_id == "":
            break
        season = int(batter_id[8:]) if len(batter_id) > 8 else int(
            input("Enter the season you would like to obtain predictions for: ").lower().strip())
        if season == "":
            continue

        for key in args.outputs:
            if key in maps:
                values = maps[key][batter_id[:8]][str(season)]
                print(key + ":")
                print(dict(zip(fields[key], values)) if key in fields else list(values))

        if "out_mean_covariance" in maps:
            mean_covariance = maps["out_mean_covariance"][batter_id[:8]][str(season)]
            mean = mean_covariance[:8]
            covariance = la.inv(mean_covariance[8:].reshape((8, 8)))
            while True:
                stat1 = input("Enter the first stat index you would like to visualize with the Gaussian Visualizer: ")
                if stat1 == "":
                    break
                stat2 = input("Enter the first stat index you would like to visualize with the Gaussian Visualizer: ")
                if stat2 == "":
                    break
                try:
                    stat1, stat2 = int(stat1), int(stat2)
                except ValueError:
                    break

                this_mean = mean[[stat1, stat2]]
                this_cov = covariance[[stat1, stat2]][:, [stat1, stat2]]
                GaussianVisualizer.plot(this_mean, this_cov, np.array([-3, -3]), np.array([3, 3]))


if args.train:
    train()
elif args.generate:
    generate()
elif args.upload:
    upload()
