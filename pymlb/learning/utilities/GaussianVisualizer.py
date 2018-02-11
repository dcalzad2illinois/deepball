import matplotlib.pyplot as plt
import numpy as np


class GaussianVisualizer:
    # from https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    @staticmethod
    def plot(mean: np.ndarray, covariance: np.ndarray, axes_min: np.ndarray, axes_max: np.ndarray):
        assert axes_min.ndim == 1
        assert axes_max.ndim == 1
        assert mean.ndim == 1
        assert covariance.ndim == 2
        assert axes_min.shape == axes_max.shape
        assert mean.shape == axes_min.shape
        assert covariance.shape[0] == mean.shape[0]
        assert covariance.shape[0] == covariance.shape[1]

        # for now, force them to use 2D only
        assert mean.shape[0] == 2

        # Our 2-dimensional distribution will be over variables X and Y
        n = 60
        x, y = np.meshgrid(np.linspace(axes_min[0], axes_max[0], n), np.linspace(axes_min[1], axes_max[1], n))

        def multivariate_gaussian(pos):
            n = mean.shape[0]
            sigma_det = np.linalg.det(covariance)
            sigma_inv = np.linalg.inv(covariance)
            normalization = np.sqrt((2 * np.pi) ** n * sigma_det)
            # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mean, sigma_inv, pos - mean)

            return np.exp(-fac / 2) / normalization

        # evaluate kernels at grid points
        xy = np.c_[x.ravel(), -y.ravel()]
        zz = multivariate_gaussian(xy).reshape((n, n))

        plt.imshow(zz, extent=[axes_min[0], axes_max[0], axes_min[1], axes_max[1]], interpolation="bicubic")
        plt.show()
