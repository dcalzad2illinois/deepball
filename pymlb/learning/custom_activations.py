import keras.backend as K
from math import sqrt


def gaussian(x):
    # the last dimensions of y_true and y_pred should be n * (n + 1), where the first 'n' elements correspond to the
    # means and the last n*n elements are the entries in the covariance matrix (row-wise)

    # the results will be [mean, flattened precision matrix]

    shape = K.int_shape(x)
    n = int((sqrt(4 * shape[-1] + 1) - 1) / 2)

    # flatten both
    x_flat = K.reshape(x, (-1, shape[-1]))

    # find the predicted mean and variance for both y_true and y_pred
    mean = K.reshape(x_flat[:, :n], (-1, n))
    cov_a = K.reshape(x_flat[:, n:], (-1, n, n))

    # compute (cov_a) (cov_a^T) + I
    precision = K.batch_dot(K.permute_dimensions(cov_a, (0, 2, 1)), cov_a, axes=[1, 2]) + K.expand_dims(K.eye(n), axis=0)

    # merge them together
    merged = K.concatenate([mean, K.reshape(precision, (-1, n * n))], axis=-1)

    # un-flatten it
    return K.reshape(merged, K.shape(x))
