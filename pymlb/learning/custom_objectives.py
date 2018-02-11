import keras.backend as K
from keras.objectives import mean_absolute_error
from math import sqrt
import tensorflow as tf
import numpy as np


def mae_variance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    std_pred = K.var(y_pred, axis=-2, keepdims=True)
    std_true = K.var(y_true, axis=-2, keepdims=True)

    return mae + 10 * K.sum(K.abs(std_true - std_pred), axis=-1)


def percentile_loss(y_true, y_pred, pct: float = 0.5):
    return 2 * K.mean(pct * K.relu(y_pred - y_true) + (1.0 - pct) * K.relu(y_true - y_pred), axis=-1)


def outlier_loss(y_true, y_pred, outlier_weight: float = 0.5):
    return K.mean(K.square(y_true - y_pred) * (1 + outlier_weight * K.abs(y_true)), axis=-1)


def mse_relative(y_true, y_pred):
    diff = K.square((y_true - y_pred) / K.std(K.reshape(y_true, (-1, K.int_shape(y_pred)[-1])), axis=0, keepdims=False))
    return K.mean(diff, axis=-1)


def mae_relative(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.std(K.reshape(y_true, (-1, K.int_shape(y_pred)[-1])), axis=0, keepdims=False))
    return K.mean(diff, axis=-1)


def mse_weighted(y_true, y_pred, weight_index: int = -1, weight_mult: float = 700):
    # compute the sample weights
    sample_weights = y_true[:, :, weight_index] / weight_mult

    # find the differences
    diff = K.square(y_true - y_pred)

    # find the differences with the exception of the weight index
    if weight_index == -1:
        nonweighted_diff = diff[:, :, :weight_index]
    elif weight_index == 0:
        nonweighted_diff = diff[:, :, weight_index + 1:]
    else:
        nonweighted_diff = K.concatenate([diff[:, :, :weight_index], diff[:, :, weight_index + 1:]], axis=-1)

    # weight the differences by the sample weights
    weighted_diff = nonweighted_diff * K.expand_dims(sample_weights)

    # add back the weight item
    weighted_diff = K.concatenate([weighted_diff, K.expand_dims(diff[:, :, weight_index])], axis=-1)

    return K.mean(weighted_diff, axis=-1)


# from https://stackoverflow.com/questions/44194063/calculate-log-of-determinant-in-tensorflow-when-determinant-overflows-underflows

# from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 100000000))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_grad.py
# Gradient for logdet
def logdet_grad(op, grad):
    a = op.inputs[0]
    a_adj_inv = tf.matrix_inverse(a, adjoint=True)
    out_shape = tf.concat([tf.shape(a)[:-2], [1, 1]], axis=0)
    return tf.reshape(grad, out_shape) * a_adj_inv


# define logdet by calling numpy.linalg.slogdet
def logdet(a, name=None):
    with tf.name_scope(name, 'LogDet', [a]) as name:
        def callback(a_array):
            sign, ld = np.linalg.slogdet(a_array)
            if np.any(sign <= 0):
                print(a_array)
                raise ValueError("oops!")
            else:
                return ld

        res = py_func(callback,
                      [a],
                      tf.float32,
                      name=name,
                      grad=logdet_grad)  # set the gradient
        res.set_shape(a.get_shape()[:-2])
        return res


def gaussian_loss(y_true, y_pred):
    # the last dimensions of y_true and y_pred should be n * (n + 1), where the first 'n' elements correspond to the
    # means and the last n*n elements are the entries in the covariance matrix (row-wise)

    shape = K.int_shape(y_pred)
    n = int((sqrt(4 * shape[-1] + 1) - 1) / 2)

    # flatten both
    y_true_flat = K.reshape(y_true, (-1, shape[-1]))
    y_pred_flat = K.reshape(y_pred, (-1, shape[-1]))

    # find the predicted mean and variance for both y_true and y_pred
    def get_mean_precision(tensor):
        mean = K.reshape(tensor[:, :n], (-1, n))
        precision = K.reshape(tensor[:, n:], (-1, n, n))

        return mean, precision

    true_value, _ = get_mean_precision(y_true_flat)
    mean, precision = get_mean_precision(y_pred_flat)

    # Likelihood = ((2pi)^k * |cov|)^(-1/2) * exp(-1/2 * (x - u)^T (cov^-1) (x - u))
    # Log likelihood = -1/2 log (2pi)^k - 1/2 log |cov| - 1/2 * (x - u)^T (cov^-1) (x - u)
    # = 1/2 (-k log 2pi - log |cov| - (x - u)^T (cov^-1) (x - u)) (removing constants)
    # = 1/2 (-k log 2pi + log |cov^-1| - (x - u)^T (cov^-1) (x - u)) (substituted in cov^-1 for cov)
    # STOP here when they are using the gaussian activation (except flip the signs)
    # Since cov^-1 is symmetric, let cov^-1 = cov_A^T cov_A + epsilon I.
    # = 1/2 (-k log 2pi + log |cov_A^T cov_A + epsilon I| - (x - u)^T (cov_A)^T (cov_A) (x - u) - epsilon * (x - u)^T (x - u))
    # = 1/2 (-k log 2pi + log |cov_A^T cov_A + epsilon I| - ||cov_A (x - u)||^2 - epsilon ||x - u||^2)
    # We want to maximize this quantity, so we want to minimize
    # = 1/2 (k log 2pi + ||cov_A (x - u)||^2 + epsilon ||x - u||^2 - log |cov_A^T cov_A + epsilon I|)
    # Therefore, we will learn the cov_A and u that minimize this quantity

    # epsilon = 1
    # losses = n * np.log(2 * np.pi) + \
    #          K.sum(K.square(K.batch_dot(true_value - mean, cov_a, axes=[1, 2])), axis=-1) \
    #          + K.sum(epsilon * K.square(true_value - mean), axis=-1) \
    #          - logdet(K.batch_dot(cov_a, K.permute_dimensions(cov_a, (0, 2, 1)), axes=[1, 2]) + K.expand_dims(epsilon * K.eye(n),axis=0))

    losses = n * np.log(2 * np.pi) - K.expand_dims(logdet(precision), axis=-1) + \
             K.reshape(K.batch_dot(K.batch_dot(precision, K.expand_dims(true_value - mean, axis=1), axes=[1, 2]),
                         K.expand_dims(true_value - mean, axis=1), axes=[1, 2]), (-1, 1))

    # un-flatten it
    return 0.5 * K.reshape(losses, K.shape(y_pred)[:-1])
