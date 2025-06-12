import numpy as np
from tqdm import tqdm
import GPy
from model.AcquisitionFunction import AcquisitionFunction
from model.ClassificationRule import ClassificationRule
import time


class BaseLevelSetEstimation(object):
    def __init__(self, obj_func, kernel, acq_func_name, rule_name, stopping_criteria, params, init_sample_size, n_subsample=None, mean_function=None):
        self.kernel = kernel.copy()
        self.stopping_criteria = stopping_criteria
        self.obj_func = obj_func
        self.eval_points = obj_func.candidate_points
        self.n_eval_points = obj_func.candidate_points.shape[0]
        self.n_subsample = n_subsample
        self.threshold = obj_func.threshold
        self.params = params
        self.init_sample_size = init_sample_size
        self.acq_func = AcquisitionFunction(acq_func_name, obj_func.threshold, **params)
        self.cls_rule = ClassificationRule(rule_name, obj_func.threshold, obj_func.candidate_points, **params)
        self.mean_function = mean_function
        self.prior_len = None
        self.prior_var = None

        self.history = {}
        self.history["length"] = []
        self.history["var"] = []
        self.history["noise_var"] = []
        self.history["pos"] = []
        self.history["upper_set"] = []
        self.history["lower_set"] = []
        self.history["upper_set_f_score"] = []
        self.history["lower_set_f_score"] = []
        self.history["unclassified_set"] = []
        self.history["pred_class"] = []

    def explore(self, n_explore):
        for sc in self.stopping_criteria:
            sc.reset(n_explore)
        for i in tqdm(range(1, n_explore+1)):
            gp = self.train_gp(self.X, self.y, self.prior_len, self.prior_var)
            self.cls_rule.classify_candidates(gp)

            self.candidate_points, self.current_active_set = self.get_current_candidate_point(self.eval_points, self.cls_rule.active_set)
            sampled_arms, a = self.acq_func.get_next_data(gp, self.candidate_points, active_set=self.current_active_set, n_eval_points=self.n_eval_points)
            x_new = self.candidate_points[sampled_arms]

            self.save_history(gp)

            if self.check_stopping_condition(gp, i):
                self.set_remain_history(i, n_explore)
                break

            self.update_dataset(x_new)

        gp = self.train_gp(self.X, self.y, self.prior_len, self.prior_var)
        self.cls_rule.classify_candidates(gp)
        self.save_history(gp)

        self.check_stopping_condition(gp, i)

        # set number of iteration as a stopping time when stopping condition is not satisfied
        for sc in self.stopping_criteria:
            sc.set_budget_timing(n_explore)

    def set_prior(self, mean, var):
        b = mean / var
        a = mean * b
        return GPy.priors.Gamma(a, b)

    def train_gp(self, X, y, prior_len, prior_var):
        if self.mean_function is not None:
            mf = GPy.core.Mapping(self.eval_points.shape[1], 1)
            mf.f = lambda x: self.mean_function
            mf.update_gradients = lambda a, b: None
        else:
            mf = None
        gp = GPy.models.GPRegression(X, y, kernel=self.kernel, mean_function=mf)
        if prior_len is not None:
            gp.kern.lengthscale.set_prior(prior_len, warning=False)
        if prior_var is not None:
            gp.kern.variance.set_prior(prior_var, warning=False)
        gp.likelihood.variance.constrain_bounded(1e-6, 1e+6, warning=False)
        gp.optimize()
        self.kernel = gp.kern.copy()
        return gp

    def get_current_candidate_point(self, eval_points, active_set):
        return eval_points, active_set

    def check_stopping_condition(self, gp, current_time):
        for sc in self.stopping_criteria:
            if sc.name == "FC":
                sc.check_threshold(len(self.cls_rule.unclassified_set), self.n_eval_points, current_time)
            elif sc.name == "Ours":
                sc.check_threshold(gp, self.eval_points, self.threshold, current_time)
            elif sc.name == "MCSampling":
                sc.check_threshold(gp, self.candidate_points, self.threshold, current_time)
            elif sc.name == "F1Sampling":
                sc.check_threshold(gp, self.candidate_points, self.threshold, current_time)

        # In this code, LSE is not stopped except for MELK in order to also investigate the behavior after stopping.
        if self.acq_func.acq_func_name == "MELK":
            return self.cls_rule.active_set.size == 0
        else:
            return False

    def save_history(self, gp):
        self.history["pos"].append(self.cls_rule.pos)
        self.history["length"].append(self.kernel.lengthscale[0])
        self.history["var"].append(self.kernel.variance[0])
        self.history["noise_var"].append(gp.likelihood.variance[0])
        self.history["upper_set"].append(self.cls_rule.upper_set)
        self.history["lower_set"].append(self.cls_rule.lower_set)
        self.history["upper_set_f_score"].append(self.cls_rule.upper_set_f_score)
        self.history["lower_set_f_score"].append(self.cls_rule.lower_set_f_score)
        self.history["unclassified_set"].append(self.cls_rule.unclassified_set)
        self.history["pred_class"].append(self.cls_rule.pred_class)

    def set_remain_history(self, current_time, budget):
        post_time = budget - current_time
        for key in self.history.keys():
            for i in range(1, post_time+1):
                self.history[key].append(self.history[key][-1])
        for sc in self.stopping_criteria:
            for i in range(1, post_time+1):
                if sc.name == "Ours" or sc.name == "MCSampling":
                    sc.seq_values = np.append(sc.seq_values, sc.seq_values[-1])
                if sc.name == "Ours":
                    for key in sc.history.keys():
                        sc.history[key].append(sc.history[key][-1])

    def update_dataset(self, x_new):
        y_new = self.get_new_output(x_new)
        self.X = np.concatenate([self.X, x_new], axis=0)
        self.y = np.concatenate([self.y, y_new], axis=0)

    def get_initial_sample(self, init_sample_size=10):
        pass

    def get_new_output(self, x_new):
        pass


class TestFunctionLSE(BaseLevelSetEstimation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.y = self.get_initial_sample(self.init_sample_size)
        pri_len_mean = 0.1 * (self.obj_func.x_range.max() - self.obj_func.x_range.min())
        pri_len_var = 0.1
        self.prior_len = self.set_prior(pri_len_mean, pri_len_var)
        pri_var_mean = self.y.std()**2
        pri_var_var = 0.1
        self.prior_var = self.set_prior(pri_var_mean, pri_var_var)

    def get_new_output(self, x_new):
        y_new = np.array(self.obj_func.get_output(x_new))[:,None]
        return y_new

    def get_initial_sample(self, init_sample_size=10):
        X, y = self.obj_func.get_dataset(init_sample_size)
        return X, y


class LifetimeDataLSE(BaseLevelSetEstimation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.y = self.get_initial_sample(self.init_sample_size)
        pri_len_mean = 0.1 * (self.obj_func.candidate_points.max() - self.obj_func.candidate_points.min())
        pri_len_var = 0.1
        self.prior_len = self.set_prior(pri_len_mean, pri_len_var)
        pri_var_mean = self.y.std()**2
        pri_var_var = 0.1
        self.prior_var = self.set_prior(pri_var_mean, pri_var_var)

    def get_new_output(self, x_new):
        return self.obj_func.get_output(x_new)

    def get_initial_sample(self, init_sample_size=10):
        X, y = self.obj_func.get_dataset(init_sample_size)
        return X, y

    def get_current_candidate_point(self, eval_points, active_set):
        candidate_set = self.obj_func.remain_indecies
        active_set = np.array(list(set(active_set) - set(self.obj_func.ignore_indecies)), dtype=int)
        if self.n_subsample is None:
            current_candidate_set = candidate_set
        else:
            if active_set.size >= self.n_subsample:
                # random sampling from active_set
                idx = np.random.choice(active_set.size, size=self.n_subsample, replace=False)
                current_candidate_set = active_set[idx]
            else:
                # Random sampling of all items in the active_set, excluding those that are not in the active_set
                not_active_set = np.array(list(set(candidate_set) - set(active_set)))
                remain_size = int(self.n_subsample - active_set.size)
                idx = np.random.choice(not_active_set.size, size=remain_size, replace=False)
                current_candidate_set = np.array(list(set(active_set) | set(not_active_set[idx])))
        selected_points = eval_points[current_candidate_set]
        current_active_set = np.nonzero(np.isin(current_candidate_set, active_set))[0]
        return selected_points, current_active_set
