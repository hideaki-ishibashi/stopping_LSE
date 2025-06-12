import numpy as np
from scipy.stats import norm
from utils import utils
import time


class AcquisitionFunction(object):
    def __init__(self, acq_func_name, th, **kwargs):
        self.acq_func_name = acq_func_name
        self.th = th
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_next_data(self, gp, candidate_points, **kwargs):
        noise_var = gp.likelihood.variance[0]
        n_candidate_points = candidate_points.shape[0]
        candidate_idx = np.arange(n_candidate_points, dtype=np.int32)
        if self.acq_func_name == "MILE":
            mean, Sig = gp.predict_noiseless(candidate_points, full_cov=True)
            mean = mean.flatten()
            var = np.diag(Sig)
            var = var.flatten()
            temp1 = var[None, :] - Sig ** 2 / (var[:, None] + noise_var)
            temp2 = np.sqrt(var[:, None] + noise_var) / np.abs(Sig) * (mean[None, :] - self.beta * np.sqrt(temp1) - self.th)
            temp0 = norm.cdf(temp2).sum(axis=1)
            ci = self.beta * np.sqrt(var)
            upper_i = np.where((mean - ci) > self.th)
            a = np.array(temp0 - len(upper_i))
            sampled_arms = [candidate_idx[np.argmax(a)]]
        elif self.acq_func_name == "RMILE":
            mean, Sig = gp.predict_noiseless(candidate_points, full_cov=True)
            mean = mean.flatten()
            var = np.diag(Sig)
            var = var.flatten()
            temp1 = var[None, :] - Sig ** 2 / (var[:, None] + noise_var)
            temp2 = np.sqrt(var[:, None] + noise_var) / np.abs(Sig) * (mean[None, :] - self.beta * np.sqrt(temp1) - self.th)
            temp0 = norm.cdf(temp2).sum(axis=1)
            ci = self.beta * np.sqrt(var)
            upper_i = np.where((mean - ci) > self.th)
            a = np.max(
                np.concatenate([np.array(temp0 - len(upper_i))[None, :], self.nu * np.sqrt(var[None, :])],
                               axis=0),
                axis=0)
            sampled_arms = [candidate_idx[np.argmax(a)]]
        elif self.acq_func_name == "Str":
            mean, var = gp.predict_noiseless(candidate_points, full_cov=False)
            mean = mean.flatten()
            var = var.flatten()
            a = self.beta * np.sqrt(var) - np.abs(mean - self.th)
            sampled_arms = [candidate_idx[np.argmax(a)]]
        elif self.acq_func_name == "US":
            mean, var = gp.predict_noiseless(candidate_points, full_cov=False)
            var = var.flatten()
            a = self.beta * np.sqrt(var)
            sampled_arms = [candidate_idx[np.argmax(a)]]
        elif self.acq_func_name == "MELK":
            active_set = kwargs["active_set"]
            K = gp.kern.K(candidate_points, candidate_points)
            if active_set.size == 0:
                a = np.ones(n_candidate_points) / n_candidate_points
            else:
                a = utils.optimize_lambda(active_set, K, num_iters=self.num_iters,
                                                step_size=self.step_size)
            idx = np.random.choice(n_candidate_points, size=self.batch_size, p=a)
            sampled_arms = candidate_idx[idx]
        elif self.acq_func_name == "Ours":
            mean, var = gp.predict_noiseless(candidate_points, full_cov=False)
            mean = mean.flatten()
            var = var.flatten()
            self.margin = utils.calc_margin(gp, self.L, self.delta, kwargs["n_eval_points"], self.margin_origin)
            p_sup = norm.cdf((mean - self.th) / np.sqrt(var))
            p_sub = norm.cdf((self.th - mean) / np.sqrt(var))
            p_unclassify = 1 - norm.cdf((self.th + self.margin - mean) / np.sqrt(var)) + norm.cdf(
                (self.th - self.margin - mean) / np.sqrt(var))
            a = np.min(np.concatenate([p_sup[None, :], p_sub[None, :], p_unclassify[None, :]], axis=0), axis=0)
            sampled_arms = [candidate_idx[np.argmax(a)]]
        else:
            a = np.random.rand(n_candidate_points)
            sampled_arms = [candidate_idx[np.argmax(a)]]
        return sampled_arms, a
