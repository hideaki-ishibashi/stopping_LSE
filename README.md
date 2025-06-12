# Code for An $(\epsilon,\delta)$-accurate level set estimation with a stopping criterion.

## Installation
Our code uses Python3.10.2 and the following packages:
- gpy             1.13.2  The Gaussian Process Toolbox
- dill            0.3.9   Lightweight pipelining with Python functions
- matplotlib      3.10.0  Python plotting package
- numpy           1.26.4  NumPy is the fundamental package for array computing with Python.
- scikit-learn    1.6.1   A set of python modules for machine learning and data mining
- tqdm            4.67.1  Fast, Extensible Progress Meter
- scipy           1.12.0  SciPy: Scientific Library for Python

## A brief overview of construction of our code

- `run_lse_test_func_main.py`
  - main code for LSE search of test functions
- `run_lse_lifetime.py`
  - main code for LSE search of lifetime dataset
- `calc_prob_ineq.py`
  - main code for the experiment of Appendix C.1
- `run_lse_test_func_change_n_candidate.py`
  - main code for the experiment of Appendix C.2
- `run_lse_test_func_change_noise.py`
  - main code for the experiment of Appendix C.5
- `run_lse_test_func_change_margin.py`
  - main code for the experiment of Appendix C.6
- `run_lse_test_func_change_threshold.py`
  - main code for the experiment of Appendix C.7
- `model` folder
  - `AcquisitionFunction.py`
    - code defining the acquisition functions
  - `ClassificationRule.py`
    - code defining the classification rules
  - `LevelSetEstimation.py`
    - code modifying the GPyOpt package
  - `stopping_criteria.py`
    - code defining the stopping criterion
- `utils` folder
  - `exp_setting.py`
    - class for storage of experimental setting and results.
  - `utils.py`
    - utilities for those other than GPyOpt
- `get_dataset`
    - `dataset` folder
      - `lifetime_data`
        - data3.txt
          - data file of liftetime1 in our paper
        - data4.txt
          - data file of liftetime2 in our paper
    - `test_function.py`
      - code loading various test function
    - `get_lifetime_data.py`
      - code loading lifetime dataset

## Usage
- `LevelSetEstimation.py`
    - Level set estimation is implemented by using GPy.
      - How to use GPy : https://github.com/SheffieldML/GPy
    - When definning the level set estimation, the following arguments are required.
      - `obj_func`
        - Class of target function
      - `acq_func_name`
        - Name of acuisition function
        - You can choice of `MELK`, `MILE`, `RMILE`, `Straddle`, `US`, `Ours`.
      - `kernel`
        - Kernel function of GP which is defined as `GPy.kern`.
      - `rule_name`
        - Name of acuisition function.
        - You can choice of `Ours`, `ConfidenceBound`, and `MELK`.
      - `stopping_criteria`
        - List of the stopping criterion
        - Stopping criteria is defined in `stopping_criteria.py`
      - `threshold`
        - Threshold of LSE
      - `params`
        - A dictionary for storage parameters of each acquisition function
      - `init_sample_size`
        - initial sample size. 
      - `n_subsample`
        - Size of candidate points to be explored in each iteration.
        - Candidate points are selected randomly.
        - When `n_subsample` is `None`, LSE explores the whole candidate points in each iteration.
      - `mean_function`
        - Mean function of GP prior
        - When `mean_function` is `None`, mean function of GP prior is set to zero.
    - `TestFunctionLSE` class is the class used to execute LSE for the test function
    - `LifetimeDataLSE` class is the class used to execute LSE for the lifetime dataset
- `stopping_criteria.py`
    - When definning the stopping criterion, threshold and budget are required.
    - check_threshold function calculates a value of each stopping criterion and determines if the threshold has fallen below a threshold.
