import numpy as np
from scipy.stats import norm
from utils import utils


class ClassificationRule(object):
    def __init__(self, rule_name, th, eval_points, **kwargs):
        self.rule_name = rule_name
        self.th = th
        self.upper_set = []
        self.lower_set = []
        self.eval_points = eval_points
        self.n_eval_points = eval_points.shape[0]
        self.active_set = np.arange(self.n_eval_points)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def classify_candidates(self, gp):
        mean, var = gp.predict_noiseless(self.eval_points, full_cov=False)
        mean = mean.flatten()
        var = var.flatten()
        self.pos = [mean, var]
        if self.rule_name == "Ours":
            self.margin = utils.calc_margin(gp, self.L, self.delta, self.n_eval_points, self.margin_origin)
            p_sup = norm.cdf((mean - self.th) / np.sqrt(var))
            p_sub = norm.cdf((self.th - mean) / np.sqrt(var))
            p_unclassify = norm.cdf((self.th + self.margin - mean) / np.sqrt(var)) - norm.cdf((self.th - self.margin - mean) / np.sqrt(var))
            self.classified_index = np.argmax(np.concatenate([p_sup[None, :], p_sub[None, :], p_unclassify[None, :]], axis=0), axis=0)

            # classify candidate points
            self.upper_set = np.arange(self.n_eval_points)[self.classified_index==0]
            self.lower_set = np.arange(self.n_eval_points)[self.classified_index==1]
            self.unclassified_set = np.arange(self.n_eval_points)[self.classified_index==2]
            self.upper_set_f_score = np.concatenate(
                [self.upper_set, self.unclassified_set[np.where(mean[self.unclassified_set] > self.th)[0]]],
                axis=0)
            self.lower_set_f_score = np.concatenate(
                [self.lower_set, self.unclassified_set[np.where(mean[self.unclassified_set] <= self.th)[0]]],
                axis=0)
        elif self.rule_name == "ConfidenceBound":
            ci = self.beta * np.sqrt(var)
            # classify candidate points
            self.upper_set = np.where((mean - ci) > self.th)[0]
            self.lower_set = np.where((mean + ci) <= self.th)[0]
            self.unclassified_set = np.array(list(set(np.arange(self.n_eval_points)) - set(self.upper_set) - set(self.lower_set)))
            if len(self.unclassified_set) != 0:
                self.upper_set_f_score = np.concatenate([self.upper_set, self.unclassified_set[np.where(mean[self.unclassified_set] > self.th)[0]]], axis=0)
                self.lower_set_f_score = np.concatenate([self.lower_set, self.unclassified_set[np.where(mean[self.unclassified_set] <= self.th)[0]]], axis=0)
            else:
                self.upper_set_f_score = self.upper_set
                self.lower_set_f_score = self.lower_set
        elif self.rule_name == "MELK":
            ci = self.beta * np.sqrt(var)
            # classify candidate points and remove classified candidate points
            current_upper = np.where((mean - ci) > self.th)[0]
            current_lower = np.where((mean + ci) <= self.th)[0]
            current_upper = np.array([k for k in self.active_set if k in current_upper], dtype=np.int32)
            current_lower = np.array([k for k in self.active_set if k in current_lower], dtype=np.int32)
            self.active_set = np.array([k for k in self.active_set if k not in current_upper])
            self.active_set = np.array([k for k in self.active_set if k not in current_lower])
            self.unclassified_set = self.active_set
            if len(self.upper_set) == 0:
                self.upper_set = current_upper
            else:
                self.upper_set = np.concatenate([self.upper_set, current_upper])
            if len(self.lower_set) == 0:
                self.lower_set = current_lower
            else:
                self.lower_set = np.concatenate([self.lower_set, current_lower])
            if self.active_set.size == 0:
                self.upper_set_f_score = self.upper_set
                self.lower_set_f_score = self.lower_set
            else:
                self.upper_set_f_score = np.concatenate([self.upper_set, self.unclassified_set[np.where(mean[self.unclassified_set] > self.th)[0]]], axis=0)
                self.lower_set_f_score = np.concatenate([self.lower_set, self.unclassified_set[np.where(mean[self.unclassified_set] <= self.th)[0]]], axis=0)
        self.pred_class = np.zeros(self.n_eval_points)
        self.pred_class[self.upper_set_f_score] = 1
        self.pred_class[self.lower_set_f_score] = 0

    def remove_ignore_idx(self, ignore_idx):
        self.active_set = np.array(list(set(self.active_set) - set(ignore_idx)))
