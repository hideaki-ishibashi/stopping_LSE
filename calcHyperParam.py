import GPy
from get_dataset.test_func import *


def f(X):
    return np.sin(np.pi*X)


def create_data(training_size,sigma):
    X = 2*(np.random.rand(training_size,1)-0.5)
    y = f(X[:,0])+np.random.normal(0,sigma,X.shape[0])
    return [X,y]


func_name = "ackley"
# func_name = "hartman6D"
# func_name = "wing_weight"
# th = -0.1
th = 21
noise_std = 0.0
resolution = 10
sample_size = 5000
target = TestFunctionHighDim(function_name=func_name, threshold=th, noise_std=noise_std, candidate_size=sample_size)
X = target.get_candidate(sample_size)
y = target.function(X)[:, None]
print(X.shape)
print(X.min())
print(X.max())
print(y.min())
print(y.max())
print(y[y>th].shape)
print(y[y<=th].shape)
kernel = GPy.kern.RBF(X.shape[1], variance=1, lengthscale=10, ARD=True)
mf = GPy.core.Mapping(X.shape[1], 1)
mf.f = lambda x: th
mf.update_gradients = lambda a, b: None
gp = GPy.models.GPRegression(X, y, kernel=kernel, mean_function=mf)
gp.optimize()

print(gp.kern)
print(gp.kern.lengthscale)
print(gp.Gaussian_noise.variance)
#
#   rbf.         |               value  |  constraints  |  priors
#   variance     |    6.47827245910733  |      +ve      |
#   lengthscale  |  20.139639932190835  |      +ve      |
# 0.16235791864937907
#  ackley
#   rbf.         |                value  |  constraints  |  priors
#   variance     |  0.32922309840994063  |      +ve      |
#   lengthscale  |                 (4,)  |      +ve      |
#   index  |  GP_regression.rbf.lengthscale  |  constraints  |  priors
#   [0]    |                    14.55653052  |      +ve      |
#   [1]    |                    13.96059675  |      +ve      |
#   [2]    |                    14.47735909  |      +ve      |
#   [3]    |                    14.66433131  |      +ve      |
#   index  |  GP_regression.Gaussian_noise.variance  |  constraints  |  priors
#   [0]    |                             0.15349873  |      +ve      |


# Hertmann6D
#   rbf.         |                 value  |  constraints  |  priors
#   variance     |  0.053004205263385375  |      +ve      |
#   lengthscale  |   0.29524797264981895  |      +ve      |
#   index  |  GP_regression.Gaussian_noise.variance  |  constraints  |  priors
#   [0]    |                             0.00000361  |      +ve      |

# wing_weight
#   147.07833307266606
#   447.512879336263
#   rbf.         |              value  |  constraints  |  priors
#   variance     |  4944.938871933138  |      +ve      |
#   lengthscale  |              (10,)  |      +ve      |
#   index  |  GP_regression.rbf.lengthscale  |  constraints  |  priors
#   [0]    |                   206.72283667  |      +ve      |
#   [1]    |                  2374.36443279  |      +ve      |
#   [2]    |                    10.73995539  |      +ve      |
#   [3]    |                    57.29357737  |      +ve      |
#   [4]    |                   139.86750922  |      +ve      |
#   [5]    |                     2.11094329  |      +ve      |
#   [6]    |                     0.10260596  |      +ve      |
#   [7]    |                     4.96444179  |      +ve      |
#   [8]    |                  4250.34316837  |      +ve      |
#   [9]    |                     1.97451153  |      +ve      |
#   index  |  GP_regression.Gaussian_noise.variance  |  constraints  |  priors
#   [0]    |                             0.00000044  |      +ve      |