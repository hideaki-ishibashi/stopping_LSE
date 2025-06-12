import numpy as np
from scipy.stats import norm
from utils import utils


class BaseCriterion(object):
    def __init__(self, name, start_timing=10):
        self.name = name
        self.config_name = ""
        self.stop_flags = False
        self.stop_timing = None
        self.seq_values = np.empty(0, float)
        self.start_timing = start_timing
        self.history = {}

    def reset(self, n_explore):
        self.stop_flags = False
        self.stop_timing = n_explore
        self.seq_values = np.empty(0, float)


class ProposedCriterion(BaseCriterion):
    def __init__(self, threshold=0.99, alpha=1.0, start_timing=10, isRecHist=False):
        super(ProposedCriterion, self).__init__("Ours", start_timing)
        self.threshold = threshold
        self.alpha = alpha
        self.isRecHist = isRecHist
        if self.isRecHist:
            self.history["p_sup"] = []
            self.history["p_sub"] = []
            self.history["p_unclassify"] = []
            self.history["r_min"] = []
            self.history["accuracy"] = []
            self.history["f_score"] = []

    def reset(self, n_explore):
        self.stop_flags = False
        self.stop_timing = n_explore
        self.seq_values = np.empty(0, float)
        if self.isRecHist:
            self.history["p_sup"] = []
            self.history["p_sub"] = []
            self.history["p_unclassify"] = []
            self.history["r_min"] = []
            self.history["accuracy"] = []
            self.history["f_score"] = []

    def check_threshold(self, gp, x, th, current_time, margin=None):
        mean, Sig = gp.predict_noiseless(x, full_cov=False)
        mean = mean.flatten()
        var = Sig.flatten()
        p_sup = norm.cdf((mean - th) / np.sqrt(var))
        p_sub = norm.cdf((th - mean) / np.sqrt(var))
        self.margin = utils.calc_margin(gp, self.alpha, self.threshold, x.shape[0], margin)
        p_unclassify = norm.cdf((th + self.margin - mean) / np.sqrt(var)) - norm.cdf((th - self.margin - mean) / np.sqrt(var))
        self.r_min = np.min(np.concatenate([p_sup[None, :], p_sub[None, :], 1-p_unclassify[None, :]], axis=0), axis=0)

        self.classified_index = np.argmax(np.concatenate([p_sup[None, :], p_sub[None, :], p_unclassify[None, :]], axis=0), axis=0)
        self.upper_set_size = self.classified_index[self.classified_index==0].shape[0]
        self.lower_set_size = self.classified_index[self.classified_index==1].shape[0]
        self.unclassified_set_size = self.classified_index[self.classified_index==2].shape[0]
        if (self.upper_set_size+self.lower_set_size+self.unclassified_set_size) == 0:
            self.f_score = 0
        else:
            self.accuracy = (self.upper_set_size+self.lower_set_size) / (self.upper_set_size+self.lower_set_size+self.unclassified_set_size)
        if (2*self.upper_set_size+self.unclassified_set_size) == 0:
            self.f_score = 0
        else:
            self.f_score = (2*self.upper_set_size) / (2*self.upper_set_size+self.unclassified_set_size)

        if self.isRecHist:
            self.history["p_sup"].append(p_sup)
            self.history["p_sub"].append(p_sub)
            self.history["p_unclassify"].append(p_unclassify)
            self.history["r_min"].append(self.r_min)
            self.history["accuracy"].append(self.accuracy)
            self.history["f_score"].append(self.f_score)
        self.seq_values = np.append(self.seq_values, 1-self.r_min.sum())
        if self.seq_values[-1] >= self.threshold and not self.stop_flags and self.start_timing < current_time:
            self.stop_timing = current_time
            print("{}: {}".format(self.name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, current_time):
        if not self.stop_flags:
            self.stop_timing = current_time
            print("{}: {}".format(self.name + self.config_name, current_time))
            self.stop_flags = True


class FullyClassifiedCriterion(BaseCriterion):
    def __init__(self, start_timing=10):
        super(FullyClassifiedCriterion, self).__init__("FC", start_timing)

    def check_threshold(self, n_unclassified_candidates, n_candidates, current_time):
        self.seq_values = np.append(self.seq_values, n_unclassified_candidates/n_candidates)
        if n_unclassified_candidates == 0 and not self.stop_flags and self.start_timing < current_time:
            self.stop_timing = current_time
            print("{}: {}".format(self.name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, current_time):
        if not self.stop_flags:
            self.stop_timing = current_time
            print("{}: {}".format(self.name + self.config_name, current_time))
            self.stop_flags = True


class MCSamplingCriterion(BaseCriterion):
    def __init__(self, sample_size=10000, threshold=0.99, alpha=1.0, start_timing=10):
        super(MCSamplingCriterion, self).__init__("MCSampling", start_timing)
        self.threshold = threshold
        self.alpha = alpha
        self.sample_size = sample_size

    def check_threshold(self, gp, x, th, current_time, margin=None):
        mean, cov = gp.predict_noiseless(x, full_cov=True)
        Sig = np.diag(cov)
        mean = mean.flatten()
        var = Sig.flatten()
        p_sup = norm.cdf((mean - th) / np.sqrt(var))
        p_sub = norm.cdf((th - mean) / np.sqrt(var))
        if margin is None:
            s = gp.kern.variance[0]
            beta = 1.0 / gp.likelihood.variance[0]
            desire_var = s / (1 + beta * s * self.alpha)
            desire_std = np.sqrt(desire_var)
            corrected_term = norm.ppf(0.5*((1 - (1-self.threshold)/x.shape[0]) + 1))
            self.margin = desire_std * corrected_term
        else:
            self.margin = margin

        p_unclassify = norm.cdf((th + self.margin - mean) / np.sqrt(var)) - norm.cdf((th - self.margin - mean) / np.sqrt(var))
        p_min = np.min(np.concatenate([p_sub[None, :], p_sup[None, :]], axis=0), axis=0)
        p_max = np.max(np.concatenate([p_sub[None, :], p_sup[None, :]], axis=0), axis=0)
        sample_path = np.random.multivariate_normal(mean, cov, self.sample_size)
        z = np.where(sample_path > th, 1, 0)
        w = np.where(np.logical_and(sample_path > th - self.margin, sample_path <= th + self.margin), 1, 0)
        a = np.abs(z - p_sup)
        p_mid = 0.5 * (p_min + p_max)
        gamma = np.max(np.concatenate([p_mid[None, :], p_unclassify[None, :]], axis=0), axis=0)
        eta = np.where(p_unclassify - p_max > 0, 1, 0)
        condition = np.prod(np.where(np.logical_and(a <= gamma, w >= eta), 1, 0), axis=1)
        prob = condition.sum() / self.sample_size
        self.seq_values = np.append(self.seq_values, prob)
        if self.seq_values[-1] >= self.threshold and not self.stop_flags and self.start_timing < current_time:
            self.stop_timing = current_time
            print("{}: {}".format(self.name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, current_time):
        if not self.stop_flags:
            self.stop_timing = current_time
            print("{}: {}".format(self.name + self.config_name, current_time))
            self.stop_flags = True


class F1SamplingCriterion(BaseCriterion):
    def __init__(self, sample_size=10000, threshold=0.99, start_timing=10):
        super(F1SamplingCriterion, self).__init__("F1Sampling", start_timing)
        self.threshold = threshold
        self.sample_size = sample_size
        self.seq_mean = np.empty(0, float)

    def check_threshold(self, gp, x, th, current_time):
        mean, cov = gp.predict_noiseless(x, full_cov=True)
        mean = mean.flatten()

        sample_path = np.random.multivariate_normal(mean, cov, self.sample_size)
        index_f = mean > th
        F1 = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            n_XF = np.count_nonzero(index_f)
            IFfXF = np.count_nonzero(sample_path[i, index_f] > th)
            IFfX = np.count_nonzero(sample_path[i] > th)
            A = IFfX + n_XF
            if A == 0:
                A = 1e-6
            F1[i] = 2*IFfXF/A
        # self.seq_values = np.append(self.seq_values, np.percentile(F1, 50))
        self.seq_values = np.append(self.seq_values, np.percentile(F1, 5))
        # print(F1.mean())
        if self.seq_values[-1] >= self.threshold and not self.stop_flags and self.start_timing < current_time:
            self.stop_timing = current_time
            print("{}: {}".format(self.name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, current_time):
        if not self.stop_flags:
            self.stop_timing = current_time
            print("{}: {}".format(self.name + self.config_name, current_time))
            self.stop_flags = True
