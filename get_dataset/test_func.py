import numpy as np
import itertools
from sklearn.metrics import f1_score, accuracy_score


class TestFunction(object):
    def __init__(self, function_name, threshold, noise_std=0.1, resolution=10):
        self.function_name = function_name
        self.threshold = threshold
        if function_name == "func1d":
            self.function = self.func1d
            self.dim = 1
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -0.5
                self.x_range[d, 1] = 5.5
        elif function_name == "sin1d":
            self.function = self.sinD
            self.dim = 1
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -2 * np.pi
                self.x_range[d, 1] = 2 * np.pi
        elif function_name == "sin":
            self.function = self.sinD
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -1.99*np.pi
                self.x_range[d, 1] = 1.99*np.pi
        elif function_name == "ackley":
            self.function = self.ackley
            self.dim = 4
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -32.768
                self.x_range[d, 1] = 32.768
            self.x_min = np.zeros((1,2))
        elif function_name == "alpine01":
            self.function = self.alpine01
            self.dim = 4
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -10
                self.x_range[d, 1] = 10
            self.x_min = np.zeros(self.dim)[None, :]
        elif function_name == "hartman3D":
            self.function = self.hartman3D
            self.dim = 3
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = 0
                self.x_range[d, 1] = 1
            self.x_min = np.array([0.114614,0.555649,0.852547])[None,:]
        elif function_name == "hartman6D":
            self.function = self.hartman6D
            self.dim = 6
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = 0
                self.x_range[d, 1] = 1
            self.x_min = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None, :]
        elif function_name == "styblinski_tang":
            self.function = self.styblinski_tang
            self.dim = 3
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -5
                self.x_range[d, 1] = 5
            self.x_min = -2.93534*np.ones(self.dim)[None,:]
        elif function_name == "bukin":
            self.function = self.bukin06
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0,0] = -15
            self.x_range[0,1] = -5
            self.x_range[1,0] = -3
            self.x_range[1,1] = 3
            self.x_min = np.array([-10,1])[None,:]
        elif function_name == "cross_in_tray":
            self.function = self.cross_in_tray
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0,0] = -10
            self.x_range[0,1] = 10
            self.x_range[1,0] = -10
            self.x_range[1,1] = 10
            self.x_min = np.array([[-1.3491,-1.3491],[1.3491,-1.3491],[-1.3491,1.3491],[1.3491,1.3491]])
        elif function_name == "drop_wave":
            self.function = self.drop_wave
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -5.12
            self.x_range[0, 1] = 5.12
            self.x_range[1, 0] = -5.12
            self.x_range[1, 1] = 5.12
            self.x_min = np.array([0,0])[None, :]
        elif function_name == "eggholder":
            self.function = self.eggholder
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -512
            self.x_range[0, 1] = 512
            self.x_range[1, 0] = -512
            self.x_range[1, 1] = 512
            self.x_min = np.array([512, 404.2319])[None,:]
        elif function_name == "holder_table":
            self.function = self.holder_table
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -10
            self.x_range[0, 1] = 10
            self.x_range[1, 0] = -10
            self.x_range[1, 1] = 10
            self.x_min = np.array([[-8.05502, -9.66459],[-8.05502, 9.66459],[8.05502, -9.66459],[8.05502, 9.66459]])
        elif function_name == "sphere":
            self.function = self.sphere
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -5.12
            self.x_range[0, 1] = 5.12
            self.x_range[1, 0] = -5.12
            self.x_range[1, 1] = 5.12
            self.x_min = np.array([0,0])[None, :]
        elif function_name == "booth":
            self.function = self.booth
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -10
            self.x_range[0, 1] = 10
            self.x_range[1, 0] = -10
            self.x_range[1, 1] = 10
            self.x_min = np.array([1, 3])[None, :]
        elif function_name == "matyas":
            self.function = self.matyas
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -10
            self.x_range[0, 1] = 10
            self.x_range[1, 0] = -10
            self.x_range[1, 1] = 10
            self.x_min = np.array([0, 0])[None, :]
        elif function_name == "six_hump_camel":
            self.function = self.six_hump_camel
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -3
            self.x_range[0, 1] = 3
            self.x_range[1, 0] = -2
            self.x_range[1, 1] = 2
            self.x_min = np.array([[0.0898, -0.7126],[-0.0898, 0.7126]])
        elif function_name == "michalewicz":
            self.function = self.michalewicz
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = 0.0
            self.x_range[0, 1] = np.pi
            self.x_range[1, 0] = 0.0
            self.x_range[1, 1] = np.pi
            self.x_min = np.array([[2.20, 1.57]])
        elif function_name == "branin":
            self.function = self.branin
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -5.0
            self.x_range[0, 1] = 10.0
            self.x_range[1, 0] = 0.0
            self.x_range[1, 1] = 15.0
            self.x_min = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        elif function_name == "easom":
            self.function = self.easom
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = -100
            self.x_range[0, 1] = 100
            self.x_range[1, 0] = -100
            self.x_range[1, 1] = 100
            self.x_min = np.array([[np.pi, np.pi]])
        elif function_name == "rosenbrock":
            self.function = self.rosenbrock
            self.dim = 2
            self.x_range = np.zeros((self.dim, 2))
            # self.x_range[:, 0] = -5
            # self.x_range[:, 1] = 10
            self.x_range[:, 0] = -3
            self.x_range[:, 1] = 3
            self.x_min = np.ones(self.dim)[None, :]
        elif function_name == "wing_weight":
            self.function = self.wing_weight
            self.dim = 10
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = 150
            self.x_range[0, 1] = 200
            self.x_range[1, 0] = 220
            self.x_range[1, 1] = 300
            self.x_range[2, 0] = 6
            self.x_range[2, 1] = 10
            self.x_range[3, 0] = -10
            self.x_range[3, 1] = 10
            self.x_range[4, 0] = 16
            self.x_range[4, 1] = 45
            self.x_range[5, 0] = 0.5
            self.x_range[5, 1] = 1
            self.x_range[6, 0] = 0.08
            self.x_range[6, 1] = 0.18
            self.x_range[7, 0] = 2.5
            self.x_range[7, 1] = 6
            self.x_range[8, 0] = 1700
            self.x_range[8, 1] = 2500
            self.x_range[9, 0] = 0.025
            self.x_range[9, 1] = 0.08
        else:
            self.function = None
        self.noise_std = noise_std
        self.candidate_points = self.get_grid(resolution)
        self.candidate_indecies = np.arange(self.candidate_points.shape[0], dtype=np.int32)
        self.env_name = f"{function_name}_{threshold}_{noise_std}"
        # self.env_name = f"{function_name}_{noise_std}"

    def func1d(self, x):
        res = (
                x[:, 0]
                + 7 * np.exp(-0.5 * np.square(x[:, 0] - 3) / np.square(1))
                + 5 * np.sin(2 * x[:, 0])
                - 7 * np.exp(-0.5 * np.square(x[:, 0] - 3.8) / np.square(0.35))
                + 1 * np.exp(-0.5 * np.square(x[:, 0] - 0.8) / np.square(0.4))
        )
        return res

    def sinD(self, X):
        y = np.sin(X).prod(axis=1)
        return y

    def ackley(self, X):
        dim = X.shape[1]
        a = 20
        b = 0.2
        c = 2*np.pi
        y = -a*np.exp(-b*np.sqrt((X**2).sum(axis=1)/dim))-np.exp(np.cos(c*X).sum(axis=1)/dim)+a+np.e
        return y

    def hartman3D(self, X):
        a = np.array([1.0,1.2,3.0,3.2])
        A = np.array([[3.0,10.0,30],
                      [0.1,10.0,35],
                      [3.0,10.0,30],
                      [0.1,10.0,35]])
        P = 1e-4*np.array([[3689.0,1170.0,2673.0],
                           [4699.0,4387.0,7470.0],
                           [1091.0,8732.0,5547.0],
                           [381.0,5743.0,8828.0]])
        y = -(a[None,:]*np.exp(-(A[None,:,:]*(X[:,None,:]-P[None,:,:])**2).sum(axis=2))).sum(axis=1)
        return y

    def hartman6D(self, X):
        a = np.array([1.0,1.2,3.0,3.2])
        A = np.array([[10.0,3.0,17.0,3.5,1.7,8],
                      [0.05,10.0,17.0,0.1,8.0,14.0],
                      [3.0,3.5,1.7,10.0,17.0,8.0],
                      [17.0,8.0,0.05,10.0,0.1,14]])
        P = 1e-4*np.array([[1312.0,1696.0,5569.0,124.0,8283.0,5886.0],
                           [2329.0,4135.0,8307.0,3736.0,1004.0,9991.0],
                           [2348.0,1451.0,3522.0,2883.0,3047.0,6650.0],
                           [4047.0,8828.0,8732.0,5743.0,1091.0,381.0]])
        y = -(a[None,:]*np.exp(-(A[None,:,:]*(X[:,None,:]-P[None,:,:])**2).sum(axis=2))).sum(axis=1)
        return y

    def styblinski_tang(self, X):
        y = 0.5*(X**4-16*X**2+5*X).sum(axis=1)
        return y

    def alpine01(self, X):
        y = np.sum(np.abs(X*np.sin(X)+0.1*X),axis=1)
        return y

    def bukin06(self, X):
        y = 100*np.sqrt(np.abs(X[:,1]-0.01*X[:,0]**2))+0.01*np.abs(X[:,0]+10)
        return y

    def cross_in_tray(self, X):
        y = -0.0001*(np.abs(np.sin(X[:,0])*np.sin(X[:,1])*np.exp(np.abs(100-np.sqrt(X[:,0]**2+X[:,1]**2)/np.pi)))+1)**0.1
        return y

    def drop_wave(self, X):
        y = -(1+np.cos(12*np.sqrt(X[:,0]**2+X[:,1]**2)))/(0.5*(X[:,0]**2+X[:,1]**2)+2)
        return y

    def eggholder(self, X):
        y = -(X[:,1]+47)*np.sin(np.sqrt(np.abs(X[:,1]+X[:,0]/2+47)))-X[:,0]*np.sin(np.sqrt(np.abs(X[:,0]-(X[:,1]+47))))
        return y

    def holder_table(self, X):
        y = -np.abs(np.sin(X[:,0])*np.cos(X[:,1])*np.exp(np.abs(1-np.sqrt((X**2).sum(axis=1))/np.pi)))
        return y

    def sphere(self, X):
        y = (X**2).sum(axis=1)
        return y

    def booth(self, X):
        y = (X[:,0]+2*X[:,1]-7)**2+(2*X[:,0]+X[:,1]-5)**2
        return y

    def matyas(self, X):
        y = 0.26*((X**2).sum(axis=1))-0.48*X[:,0]*X[:,1]
        return y

    def six_hump_camel(self, X):
        y = (4-2.1*X[:,0]**2+X[:,0]**4/3)*X[:,0]**2+X[:,0]*X[:,1]+(-4+4*X[:,1]**2)*X[:,1]**2
        return y

    def michalewicz(self, X):
        m = 10
        y = - (np.sin(X) * np.sin(np.arange(1, X.shape[1]+1) * X**2 / np.pi)**(2*m)).sum(axis=1)
        return y

    def branin(self, X):
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1 / (8.0 * np.pi)
        y = a * (X[:, 1] - b * X[:, 0] ** 2 + c * X[:, 0] - r)**2 + s * (1 - t) * np.cos(X[:, 0]) + s
        return y

    def easom(self, X):
        y = -np.cos(X[:, 0])*np.cos(X[:, 1])*np.exp(-(X[:, 0] - np.pi)**2-(X[:, 1] - np.pi)**2)
        return y

    def rosenbrock(self, X):
        y = 0
        for d in range(X.shape[1]-1):
            y += 100*(X[:, d+1] - X[:, d]**2)**2+(X[:, d] - 1)**2
        return y

    def wing_weight(self, X):
        Sw = X[:, 0]
        Wfw = X[:, 1]
        A = X[:, 2]
        LamCaps = X[:, 3] * (np.pi / 180)
        q = X[:, 4]
        lam = X[:, 5]
        tc = X[:, 6]
        Nz = X[:, 7]
        Wdg = X[:, 8]
        Wp = X[:, 9]

        fact1 = 0.036 * Sw ** 0.758 * Wfw ** 0.0035
        fact2 = (A / ((np.cos(LamCaps)) ** 2)) ** 0.6
        fact3 = q ** 0.006 * lam ** 0.04
        fact4 = (100 * tc / np.cos(LamCaps)) ** (-0.3)
        fact5 = (Nz * Wdg) ** 0.49

        term1 = Sw * Wp

        y = fact1 * fact2 * fact3 * fact4 * fact5 + term1
        # y = 0.036 * (X[:, 0] ** 0.758) * (X[:, 1] ** 0.0035) * (X[:, 2]/(np.cos(X[:, 3])**2))**0.6 * X[:, 4]**0.006 * X[:, 5] ** 0.04 * (100 * X[:, 6]/np.cos(X[:, 3]))**(-0.3) * (X[:, 7]*X[:, 8])**0.49 + X[:, 0] * X[:, 9]
        return y

    def get_grid(self, resolution):
        inducing_points = np.linspace(0, 1, resolution)
        nodes = itertools.product(inducing_points, repeat=self.dim)
        nodes = np.array(list(nodes))
        X = np.zeros((resolution**self.dim, self.dim))
        for d in range(self.x_range.shape[0]):
            X[:, d] = (self.x_range[d, 1] - self.x_range[d, 0]) * nodes[:, d] + self.x_range[d, 0]
        return X

    def uniform_sample(self, sample_size):
        X = np.random.uniform(0, 1, [sample_size, self.dim])
        for d in range(self.x_range.shape[0]):
            X[:, d] = (self.x_range[d, 1] - self.x_range[d, 0]) * X[:, d] + self.x_range[d, 0]
        return X

    def get_output(self, X):
        f = self.function(X)
        y = f + np.random.normal(0, self.noise_std, X.shape[0])
        return y

    def get_dataset(self, sample_size):
        X = np.zeros((sample_size, self.x_range.shape[0]))
        for d in range(self.x_range.shape[0]):
            X[:, d] = np.random.uniform(self.x_range[d, 0], self.x_range[d, 1], sample_size)
        f = self.function(X)[:, None]
        noise = np.random.normal(0, self.noise_std, sample_size)
        y = f + noise[:, None]
        return X, y

    def evaluate_LSE(self, pred_class):
        # calc true class
        true_func = self.function(self.candidate_points)
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[true_func - self.threshold > 0] = 1
        true_class[true_func - self.threshold <= 0] = 0

        # calc F1 score
        n_explore = len(pred_class)
        f1_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            f1_score_list[b] = f1_score(true_class, pred_class[b])

        return f1_score_list

    def evaluate_LSE_with_noise(self, pred_class):
        # calc true class
        true_func = self.get_output(self.candidate_points)
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[true_func - self.threshold > 0] = 1
        true_class[true_func - self.threshold <= 0] = 0

        # calc F1 score
        n_explore = len(pred_class)
        f1_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            f1_score_list[b] = f1_score(true_class, pred_class[b])

        return f1_score_list

    def accuracy_LSE(self, pred_class):
        # calc true class
        # true_func = self.get_output(self.candidate_points)
        true_func = self.function(self.candidate_points)
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[true_func - self.threshold > 0] = 1
        true_class[true_func - self.threshold <= 0] = 0

        # calc F1 score
        n_explore = len(pred_class)
        accuracy_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            accuracy_score_list[b] = accuracy_score(true_class, pred_class[b])

        return accuracy_score_list

    def accuracy_LSE_with_noise(self, pred_class):
        # calc true class
        # calc F1 score
        true_func = self.get_output(self.candidate_points)
        # true_func = self.function(self.candidate_points)
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[true_func - self.threshold > 0] = 1
        true_class[true_func - self.threshold <= 0] = 0

        n_explore = len(pred_class)
        accuracy_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            accuracy_score_list[b] = accuracy_score(true_class, pred_class[b])

        return accuracy_score_list


class TestFunctionHighDim(object):
    def __init__(self, function_name, threshold, noise_std=0.1, candidate_size=10000):
        self.function_name = function_name
        self.threshold = threshold
        self.function_name = function_name
        self.threshold = threshold

        if function_name == "ackley":
            self.function = self.ackley
            self.dim = 4
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = -32.768
                self.x_range[d, 1] = 32.768
            self.x_min = np.zeros((1,2))
        elif function_name == "hartman6D":
            self.function = self.hartman6D
            self.dim = 6
            self.x_range = np.zeros((self.dim, 2))
            for d in range(self.dim):
                self.x_range[d, 0] = 0
                self.x_range[d, 1] = 1
            self.x_min = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])[None, :]
        elif function_name == "wing_weight":
            self.function = self.wing_weight
            self.dim = 10
            self.x_range = np.zeros((self.dim, 2))
            self.x_range[0, 0] = 150
            self.x_range[0, 1] = 200
            self.x_range[1, 0] = 220
            self.x_range[1, 1] = 300
            self.x_range[2, 0] = 6
            self.x_range[2, 1] = 10
            self.x_range[3, 0] = -10
            self.x_range[3, 1] = 10
            self.x_range[4, 0] = 16
            self.x_range[4, 1] = 45
            self.x_range[5, 0] = 0.5
            self.x_range[5, 1] = 1
            self.x_range[6, 0] = 0.08
            self.x_range[6, 1] = 0.18
            self.x_range[7, 0] = 2.5
            self.x_range[7, 1] = 6
            self.x_range[8, 0] = 1700
            self.x_range[8, 1] = 2500
            self.x_range[9, 0] = 0.025
            self.x_range[9, 1] = 0.08
        self.noise_std = noise_std
        self.candidate_points = self.get_candidate(candidate_size)
        self.whole_size = self.candidate_points.shape[0]
        self.outputs = self.function(self.candidate_points) + np.random.normal(0, self.noise_std, self.candidate_points.shape[0])
        self.candidate_indecies = np.arange(self.candidate_points.shape[0], dtype=np.int32)
        self.env_name = f"{function_name}_{threshold}_{noise_std}"

    def ackley(self, X):
        dim = X.shape[1]
        a = 20
        b = 0.2
        c = 2*np.pi
        y = -a*np.exp(-b*np.sqrt((X**2).sum(axis=1)/dim))-np.exp(np.cos(c*X).sum(axis=1)/dim)+a+np.e
        return y

    def hartman6D(self, X):
        a = np.array([1.0,1.2,3.0,3.2])
        A = np.array([[10.0,3.0,17.0,3.5,1.7,8],
                      [0.05,10.0,17.0,0.1,8.0,14.0],
                      [3.0,3.5,1.7,10.0,17.0,8.0],
                      [17.0,8.0,0.05,10.0,0.1,14]])
        P = 1e-4*np.array([[1312.0,1696.0,5569.0,124.0,8283.0,5886.0],
                           [2329.0,4135.0,8307.0,3736.0,1004.0,9991.0],
                           [2348.0,1451.0,3522.0,2883.0,3047.0,6650.0],
                           [4047.0,8828.0,8732.0,5743.0,1091.0,381.0]])
        y = -(a[None,:]*np.exp(-(A[None,:,:]*(X[:,None,:]-P[None,:,:])**2).sum(axis=2))).sum(axis=1)
        return y

    def wing_weight(self, X):
        Sw = X[:, 0]
        Wfw = X[:, 1]
        A = X[:, 2]
        LamCaps = X[:, 3] * (np.pi / 180)
        q = X[:, 4]
        lam = X[:, 5]
        tc = X[:, 6]
        Nz = X[:, 7]
        Wdg = X[:, 8]
        Wp = X[:, 9]

        fact1 = 0.036 * Sw ** 0.758 * Wfw ** 0.0035
        fact2 = (A / ((np.cos(LamCaps)) ** 2)) ** 0.6
        fact3 = q ** 0.006 * lam ** 0.04
        fact4 = (100 * tc / np.cos(LamCaps)) ** (-0.3)
        fact5 = (Nz * Wdg) ** 0.49

        term1 = Sw * Wp

        y = fact1 * fact2 * fact3 * fact4 * fact5 + term1
        # y = 0.036 * (X[:, 0] ** 0.758) * (X[:, 1] ** 0.0035) * (X[:, 2]/(np.cos(X[:, 3])**2))**0.6 * X[:, 4]**0.006 * X[:, 5] ** 0.04 * (100 * X[:, 6]/np.cos(X[:, 3]))**(-0.3) * (X[:, 7]*X[:, 8])**0.49 + X[:, 0] * X[:, 9]
        return y

    def get_candidate(self, candidate_size):
        X = np.random.uniform(0, 1, [candidate_size, self.dim])
        candidates = np.zeros([candidate_size, self.dim])
        for d in range(self.x_range.shape[0]):
            candidates[:, d] = (self.x_range[d, 1] - self.x_range[d, 0]) * X[:, d] + self.x_range[d, 0]
        return candidates

    def get_output(self, X):
        f = self.function(X)
        y = f + np.random.normal(0, self.noise_std, X.shape[0])
        return y

    def get_dataset(self, sample_size):
        X = np.zeros((sample_size, self.x_range.shape[0]))
        for d in range(self.x_range.shape[0]):
            X[:, d] = np.random.uniform(self.x_range[d, 0], self.x_range[d, 1], sample_size)
        f = self.function(X)[:, None]
        noise = np.random.normal(0, self.noise_std, sample_size)
        y = f + noise[:, None]
        return X, y

    def evaluate_LSE(self, pred_class):
        # calc true class
        true_func = self.function(self.candidate_points)
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[true_func - self.threshold > 0] = 1
        true_class[true_func - self.threshold <= 0] = 0

        # calc F1 score
        n_explore = len(pred_class)
        f1_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            f1_score_list[b] = f1_score(true_class, pred_class[b])

        return f1_score_list

    def evaluate_LSE_with_noise(self, pred_class):
        # calc true class
        true_func = self.get_output(self.candidate_points)
        true_class = np.zeros(self.candidate_points.shape[0])
        true_class[true_func - self.threshold > 0] = 1
        true_class[true_func - self.threshold <= 0] = 0

        # calc F1 score
        n_explore = len(pred_class)
        f1_score_list = np.zeros(n_explore)
        for b in range(n_explore):
            f1_score_list[b] = f1_score(true_class, pred_class[b])

        return f1_score_list
