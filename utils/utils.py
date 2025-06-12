import numpy as np
import dill
from scipy.stats import norm


def create_grid(node_size,dim,x_range):
    x = np.linspace(x_range[0,0], x_range[0,1], node_size)[:, None]
    y = np.linspace(x_range[1,0], x_range[1,1], node_size)[:, None]
    node_x, node_y = np.meshgrid(x, y)
    grid = np.zeros((node_size, node_size, dim))
    grid[:, :, 0] = node_x
    grid[:, :, 1] = node_y
    node = grid.reshape(node_size*node_size,dim)

    return [grid,node]


def serialize(obj,save_name):
    with open(save_name, mode='wb') as f:
        dill.dump(obj, f)


def deserialize(save_name):
    with open(save_name, mode='rb') as f:
        obj = dill.load(f)
    return obj


def optimize_lambda(A_t, K, num_iters, step_size=1, gamma=.1):
    n = K.shape[0]
    samples = []  # maintain the indices of X's seen
    l = np.zeros(n)
    idx = np.random.randint(n)
    l[idx] = 1
    samples.append(idx)
    for t in range(1, num_iters):
        # find element with highest variance
        w = np.sqrt(l[samples])
        cov_ = K[np.ix_(samples, samples)] * np.outer(w, w) + gamma * np.eye(len(samples))
        inv_cov_ = np.linalg.inv(cov_)
        Ks = K[:, samples] * w  # pull out the relevant part of the kernel matrix |X| x samples
        A = Ks @ inv_cov_  # kernelize the points - |X| x samples
        var = np.diag(K) / gamma - np.squeeze(A[:, np.newaxis, :] @ Ks[:, :, np.newaxis]) / gamma
        y_idx = A_t[np.argmax(var[A_t])]

        # now compute the gradient at this point
        g = K[y_idx] - A[y_idx] @ Ks.T
        g = - g * g / (gamma * gamma)
        # decide on the direction
        idx = np.argmin(g)
        eta = step_size / (t + 2)
        lold = l
        l = (1 - eta) * l + eta * np.eye(n)[idx]
        samples = [i for i in range(n) if l[i] > 1e-6]
    return lold


def calc_margin(gp, L, delta, n_eval_points, margin=None):
    if margin is None:
        s = gp.kern.variance[0]
        _lambda = 1.0 / gp.likelihood.variance[0]
        desire_var = s / (1 + _lambda * s * L)
        desire_std = np.sqrt(desire_var)
        corrected_term = norm.ppf(0.5 * ((1 - (1 - delta) / n_eval_points) + 1))
        return desire_std * corrected_term
    else:
        return margin
