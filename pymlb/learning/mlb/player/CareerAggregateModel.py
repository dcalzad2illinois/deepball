from pymlb.learning import AggregateModel, Model
from pymlb.learning.aggregators import MSEPredictionAggregator, GaussianPredictionMerger, \
    PCAPredictionAggregator
import matplotlib.pyplot as plt
import numpy as np


class CareerAggregateModel(AggregateModel):
    def __init__(self, template_single: Model, regularization_mult: float = 1, *args, **kwargs):
        super().__init__(template=template_single,
                         key_classes={"out_stats": MSEPredictionAggregator(regularization_mult),
                                      "out_counts": MSEPredictionAggregator(regularization_mult),
                                      "out_fielding_position": MSEPredictionAggregator(regularization_mult),
                                      "magic_vector": PCAPredictionAggregator(column_normalization_order=2),
                                      "out_mean_covariance": GaussianPredictionMerger()},
                         *args, **kwargs)

    def visualize(self, to_file: str, *args, **kwargs):
        if not self.is_trained():
            raise RuntimeError("The aggregate model has not been trained, so it cannot be visualized.")

        array = self.model["aggregators"]["out_stats"]

        # make the time-axis labels
        x_labels = ["Model " + str(i) for i in range(1, array.shape[1] + 1)]
        y_labels = ["AVG", "SLG", "K%", "BB%", "OBP"]

        # Plot it out
        fig, ax = plt.subplots()
        ax.pcolor(array, cmap=plt.cm.get_cmap("Blues"), alpha=0.8)

        # Format
        fig = plt.gcf()
        fig.set_size_inches(array.shape[1] * 2 / 3 + 1, array.shape[0] * 2 / 3 + 1)

        # _put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(len(y_labels)) + 0.5, minor=False)
        ax.set_xticks(np.arange(len(x_labels)) + 0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        # ax.xaxis.tick_top()

        # Set the labels
        ax.set_xticklabels(x_labels, minor=False)
        ax.set_yticklabels(y_labels, minor=False)

        ax.grid(False)

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.3f' % array[y, x], horizontalalignment='center',
                         verticalalignment='center')
        plt.title("Sample Bagged Model Weights")
        plt.savefig(to_file)
