from subprocess import Popen
from os.path import isdir, isfile
from os import makedirs
import sys


def run():
    # train the single models
    for model_index in range(8):
        for max_year in range(2014, 2017):
            for model_type in ["deep-dense", "simple-rnn", "rf-linear"]:
                # only train one of certain models
                if model_type == "rf-linear" and model_index > 0:
                    continue
                model_extension = ".pkl" if model_type == "rf-linear" else ".h5"

                additional_switches = []
                if model_type == "rf-linear":
                    additional_switches = ["-o", "out_stats", "out_counts", "out_fielding_position", "--partition",
                                           "all"]

                for shift in range(1, 3):
                    # spawn all processes
                    processes = []
                    out_files = []

                    # compile the metadata/settings together
                    this_model = ["shift" + str(shift), "max" + str(max_year), model_type, str(model_index)]
                    for i in range(1, len(this_model)):
                        if not isdir("models/bc/" + "/".join(this_model[:i])):
                            makedirs("models/bc/" + "/".join(this_model[:i]))
                    model_file = "models/bc/" + "/".join(this_model) + model_extension
                    if isfile(model_file):
                        continue

                    log_file = "models/bc/" + "/".join(this_model) + ".txt"
                    print(model_file)

                    # spawn the process
                    f = open(log_file, "w", buffering=1)
                    p = Popen([sys.executable, "batter_career.py", "--train", "-m", model_file,
                               "--start-year", "1958", "--end-year", str(max_year), "--model-type", model_type,
                               "--shift", str(shift)] + additional_switches, stdout=f, stderr=f)
                    processes.append(p)
                    out_files.append(f)

                    # join all processes
                    for p in processes:
                        p.wait()
                        if p.returncode != 0:
                            print("Last call returned " + str(p.returncode) + ", exiting . . .", file=sys.stderr)
                            exit(1)

                    # close all outputs
                    for f in out_files:
                        f.close()

    # train the aggregates
    for max_year in range(2014, 2017):
        for model_type in ["deep-dense"]:
            for shift in range(1, 3):
                # spawn all processes
                processes = []
                out_files = []

                # compile the metadata/settings together
                this_model = ["shift" + str(shift), "max" + str(max_year), model_type, "aggregate"]
                for i in range(1, len(this_model)):
                    if not isdir("models/bc/" + "/".join(this_model[:i])):
                        makedirs("models/bc/" + "/".join(this_model[:i]))
                model_file = "models/bc/" + "/".join(this_model) + ".pkl"
                if isfile(model_file):
                    continue

                log_file = "models/bc/" + "/".join(this_model) + ".txt"
                print(model_file)

                # spawn the process
                f = open(log_file, "w", buffering=1)
                p = Popen([sys.executable, "batter_career.py", "--train", "-m", model_file,
                           "--start-year", "1958", "--end-year", str(max_year), "--model-type", model_type,
                           "--shift", str(shift), "--partition", "validation", "--aggregating",
                           "models/bc/" + "/".join(this_model[:-1]) + "/*.h5"],
                          stdout=f, stderr=f)
                processes.append(p)
                out_files.append(f)

                # join all processes
                for p in processes:
                    p.wait()
                    if p.returncode != 0:
                        print("Last call returned " + str(p.returncode) + ", exiting . . .", file=sys.stderr)
                        exit(1)

                # close all outputs
                for f in out_files:
                    f.close()

    # upload the results
    for model_type in ["rf-linear", "simple-rnn", "deep-dense"]:
        model_name = "0.pkl" if model_type == "rf-linear" else "0.h5" if model_type == "simple-rnn" else "aggregate.pkl"
        for shift in range(1, 3):
            for max_year in range(2016, 2013, -1):
                # spawn all processes
                processes = []
                out_files = []

                # compile the metadata/settings together
                this_model = ["shift" + str(shift), "max" + str(max_year), model_type, model_name]
                for i in range(1, len(this_model)):
                    if not isdir("models/bc/" + "/".join(this_model[:i])):
                        makedirs("models/bc/" + "/".join(this_model[:i]))
                model_file = "models/bc/" + "/".join(this_model)
                if not isfile(model_file):
                    print("Error: Model does not exist: " + model_file, file=sys.stderr)
                    continue

                # log_file = "models/bc/" + "/".join(this_model) + ".txt"
                # print(model_file)

                # spawn the process
                # f = open(log_file, "w", buffering=1)
                p = Popen([sys.executable, "batter_career.py", "--upload", "-m", model_file,
                           "--start-year", "1990", "--end-year", str(max_year + 1), "--model-type", model_type,
                           "--shift", str(shift), "--upload-suffix", model_type.replace("-", "_")] +
                          ([] if model_type == "rf-linear" or model_type == "simple-rnn" else ["--use-aggregate"]))
                processes.append(p)
                # out_files.append(f)

                # join all processes
                for p in processes:
                    p.wait()
                    if p.returncode != 0:
                        print("Last call returned " + str(p.returncode) + ", exiting . . .", file=sys.stderr)
                        exit(1)

                # close all outputs
                for f in out_files:
                    f.close()


if __name__ == "__main__":
    run()
