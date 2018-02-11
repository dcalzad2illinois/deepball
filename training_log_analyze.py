import argparse
from os.path import isfile
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re


def eprint(*a, **kwargs):
    print(*a, file=sys.stderr, **kwargs)


def parse_log(contents, monitor_loss):
    def get_dictionary(loss):
        return {int(epoch): float(value) for epoch, value in
                re.findall('Epoch (\d+)\/\d+[\d\D]*? ' + loss + ': ([\d\.]+)', contents)}

    training = get_dictionary(monitor_loss)
    validation = get_dictionary("val_" + monitor_loss)
    if len(training) == 0 or len(validation) == 0:
        return {}, {}, None

    # get the epoch with the lowest validation loss
    lowest_epoch = min((loss, epoch) for epoch, loss in validation.items())[1]
    return training, validation, lowest_epoch


parser = argparse.ArgumentParser(description="Analyzes a log file containing the training results")

action_group = parser.add_mutually_exclusive_group(required=True)
action_group.add_argument("--plot", action="store_true", help="Generates a plot of the given training log")
action_group.add_argument("--csv", action="store_true", help="Generates a CSV of the given training log")

parser.add_argument("-f", "--file", help="The log file to be parsed", required=True)
parser.add_argument("-l", "--loss", help="The loss name to be tracked", default="loss")
parser.add_argument("--y-max", help="The Y-axis maximum", type=float, default=None)
parser.add_argument("--y-min", help="The Y-axis minimum", type=float, default=None)
parser.add_argument("-t", "--title", default="", help="The plot title, if applicable")

args = parser.parse_args()


def plot(parsed):
    # red dashes, blue squares and green triangles
    plt.figure()
    training, = plt.plot(np.array(list(parsed[0].keys())), np.array(list(parsed[0].values())), "k")
    validation, = plt.plot(np.array(list(parsed[1].keys())), np.array(list(parsed[1].values())), "r")
    plt.legend([training, validation], ["Training Loss", "Validation Loss"])
    plt.axvline(x=parsed[2])
    plt.ylim(ymax=args.y_max, ymin=args.y_min)
    plt.title(args.title)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


def run():
    for file in glob.glob(args.file, recursive=True):
        with open(file, 'r') as content_file:
            parsed = parse_log(content_file.read(), args.loss)
        if args.plot:
            plot(parsed)


run()
