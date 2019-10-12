from __future__ import print_function, division

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import trange


class SVGD:

    def __init__(self):
        pass

    @staticmethod
    def svgd_kernel(theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return Kxy, dxkxy

    @staticmethod
    def update(x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        if debug:
            iterator = trange(n_iter, position=0, leave=True)
        else:
            iterator = range(n_iter)

        for it in iterator:
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = SVGD.svgd_kernel(theta, h=bandwidth)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad 
            if it == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta
