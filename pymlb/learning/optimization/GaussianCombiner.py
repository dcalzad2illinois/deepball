from typing import List
import numpy as np
from numpy.linalg import inv
import tensorflow as tf
from pymlb.learning.optimization import MatrixOptimizer
from pymlb.learning.custom_objectives import gaussian_loss


class GaussianCombiner(MatrixOptimizer):
    def __init__(self, model_variables: int, sub_model_count: int, *args, **kwargs):
        # for each sub-model, create an input and an output
        variables = {"c_" + str(i): np.identity(model_variables) / sub_model_count for i in range(sub_model_count)}
        inputs = ["ground_truth"] + ["results_" + str(i) for i in range(sub_model_count)]

        self.model_variables = model_variables
        self.sub_model_count = sub_model_count
        super().__init__(variables=variables, inputs=inputs, **kwargs)

    def _create_loss(self, inputs):
        examples = inputs["ground_truth"].get_shape().as_list()[0]

        # extract the means
        means = [
            inputs["results_" + str(i)][:, :self.model_variables] for i in range(self.sub_model_count)
        ]

        # extract the covariance matrices from these
        covariances = [
            tf.matrix_inverse(tf.reshape(inputs["results_" + str(i)][:, self.model_variables:],
                                         (-1, self.model_variables, self.model_variables))) for i in
            range(self.sub_model_count)
        ]

        # extract the variables and multiply all the variables by the identity matrix
        coefficients = [
            tf.reshape(
                tf.tile(tf.expand_dims(
                    inputs["c_" + str(i)] * tf.eye(self.model_variables, self.model_variables, dtype=tf.float32),
                    axis=0),
                    multiples=[examples, 1, 1]), (examples, self.model_variables, self.model_variables)) for i in
            range(self.sub_model_count)
        ]

        # take the inputs we've received and create the new mean and covariance
        new_mean = sum(
            tf.squeeze(tf.matmul(coefficients[i], tf.expand_dims(means[i], axis=-1)), axis=-1) for i in
            range(self.sub_model_count))
        new_precision = tf.matrix_inverse(sum(
            tf.matmul(coefficients[i], tf.matmul(covariances[i], coefficients[i])) for i in
            range(self.sub_model_count)))

        # merge them back together
        new_merged = tf.concat(
            [new_mean, tf.reshape(new_precision, (-1, self.model_variables * self.model_variables))], axis=-1)
        # new_merged = tf.reshape(new_merged, tf.shape(inputs["ground_truth"]))

        return tf.reduce_sum(gaussian_loss(inputs["ground_truth"], new_merged))

    def _projection(self, variables):
        # divide them by the sum
        total = sum(tf.abs(tf.diag_part(variables["c_" + str(i)])) for i in range(self.sub_model_count))
        ops = []
        for i in range(self.sub_model_count):
            ops.append(
                tf.assign(variables["c_" + str(i)], tf.diag(tf.abs(tf.diag_part(variables["c_" + str(i)])) / total)))
        return ops

    def optimize(self, sub_model_results: List[np.ndarray], ground_truths: np.ndarray, **kwargs):
        inputs = {"ground_truth": ground_truths}
        inputs.update({"results_" + str(i): sub_model_results[i] for i in range(len(sub_model_results))})
        results = self._optimize(**inputs)
        results_list = []
        for i in range(len(results)):
            results_list.append(np.diag(results["c_" + str(i)]))
        return results_list

    def evaluate(self, sub_model_results: List[np.ndarray], optimization_results):
        # turn the results into a (model count) x (entry count) tensor
        optimization_results = np.array(optimization_results)

        # turn the means into a (sample count) x (model count) x (entry count) tensor
        means = np.transpose(np.array([sub_model[:, :self.model_variables] for sub_model in sub_model_results]),
                             axes=[1, 0, 2])

        # find the new means
        new_means = np.array([np.sum(optimization_results * mean, axis=0) for mean in means])

        # turn the precisions into covariances: (sample count) x (model count) x (entry count) x (entry count) tensor
        covariances = np.transpose(np.array(
            [inv(np.reshape(sub_model[:, self.model_variables:], (-1, self.model_variables, self.model_variables))) for
             sub_model in sub_model_results]), axes=[1, 0, 2, 3])

        # find the new covariances
        new_covariances = np.array([np.sum(
            np.array([np.outer(weights, weights) * cov_model for cov_model, weights in zip(cov, optimization_results)]), axis=0)
                                    for cov in covariances])

        # convert these back to precisions
        new_precisions = np.array([inv(cov) for cov in new_covariances])

        # merge these
        output = np.concatenate([new_means, np.reshape(new_precisions, (new_precisions.shape[0], -1))], axis=-1)
        return output
