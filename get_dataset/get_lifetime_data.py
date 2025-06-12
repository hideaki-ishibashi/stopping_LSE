import numpy as np
import itertools
from sklearn.metrics import f1_score


class GetLifetimeData(object):
    def __init__(self, threshold, data_name="data3"):
        self.dim = 2
        self.data = np.loadtxt(f"get_dataset/dataset/lifetime_data/{data_name}.txt")
        self.ignore_indecies = []
        self.threshold = threshold
        self.candidate_points = self.data[:, :2]
        self.remain_indecies = np.arange(self.candidate_points.shape[0], dtype=np.int32)
        self.outputs = self.data[:, 2][:, None]
        self.whole_size = self.candidate_points.shape[0]
        self.env_name = data_name

    def get_output(self, x):
        remain_indecies_old = self.remain_indecies
        dist = ((self.candidate_points[self.remain_indecies][:, None, :] - x[None, :, :])**2).sum(axis=2)
        index = np.argmin(dist, axis=0)
        self.ignore_indecies = np.concatenate([self.ignore_indecies, index], axis=0)
        self.remain_indecies = np.array(list(set(range(self.whole_size)) - set(self.ignore_indecies)))
        return self.outputs[remain_indecies_old][index]

    def get_dataset(self, sample_size):
        indecies = np.random.choice(range(self.whole_size), sample_size, replace=False)
        self.remain_indecies = np.array(list(set(range(self.whole_size)) - set(indecies)))
        subX = self.candidate_points[indecies]
        subY = self.outputs[indecies]
        self.ignore_indecies = indecies
        return subX, subY

    def evaluate_LSE(self, pred_class):
        # calc true class
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[self.outputs[:, 0] - self.threshold > 0] = 1
        true_class[self.outputs[:, 0] - self.threshold <= 0] = 0

        # calc F1 score
        n_explore = len(pred_class)
        f1_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            f1_score_list[b] = f1_score(true_class, pred_class[b])

        return f1_score_list

