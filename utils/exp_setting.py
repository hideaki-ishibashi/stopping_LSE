import numpy as np


class ExpSetting(object):
    def __init__(self, target, acq_func_name, rule_name, budget, batch_size, n_iteration, stopping_criteria, params):
        self.acq_func_name = acq_func_name
        self.rule_name = rule_name
        self.n_iteration = n_iteration
        self.target = target
        self.budget = budget
        self.batch_size = batch_size
        self.n_explore = int(budget / batch_size)
        self.classified_ratio = np.zeros((self.n_iteration, self.n_explore+1))
        self.f1_score = np.zeros((self.n_iteration, self.n_explore+1))
        self.f1_score_with_noise = np.zeros((self.n_iteration, self.n_explore+1))
        self.accuracy_score = np.zeros((self.n_iteration, self.n_explore+1))
        self.accuracy_score_with_noise = np.zeros((self.n_iteration, self.n_explore+1))
        self.stopping_score = np.zeros((len(stopping_criteria), self.n_iteration))
        self.stopping_time = np.zeros((len(stopping_criteria), self.n_iteration))
        self.sample_span = batch_size * np.array(range(self.n_explore+1))
        self.stopping_criteria = stopping_criteria
        self.params = params
        self.lse_list = []

    def set_res_lse(self, lse):
        self.lse_list.append(lse)

