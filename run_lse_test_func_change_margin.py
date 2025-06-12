import matplotlib.pyplot as plt
import GPy
import matplotlib.patheffects as mpe
from model.stopping_criteria import *
from model.LevelSetEstimation import TestFunctionLSE
from get_dataset.test_func import TestFunction
import os
from utils import utils
from utils.exp_setting import ExpSetting

plt.rcParams["font.size"] = 28
plt.rcParams["figure.autolayout"] = True
seq_alpha = 0.3
st_alpha = 1


def set_params(acq_func_name, L, delta, margin, beta, nu, batch_size, num_iters, step_size):
    params = {}
    if acq_func_name == "Ours":
        params["L"] = L
        params["delta"] = delta
        params["margin_origin"] = margin
    elif acq_func_name == "MILE":
        params["margin"] = margin
        params["beta"] = beta
    elif acq_func_name == "RMILE":
        params["margin"] = margin
        params["nu"] = nu
        params["beta"] = beta
    elif acq_func_name == "Str":
        params["margin"] = margin
        params["beta"] = beta
    elif acq_func_name == "US":
        params["margin"] = margin
        params["beta"] = beta
    elif acq_func_name == "MELK":
        params["batch_size"] = batch_size
        params["margin"] = margin
        params["num_iters"] = num_iters
        params["step_size"] = step_size
        params["beta"] = beta
    else:
        params["margin"] = margin
    return params


if __name__ == '__main__':
    # func_name_list = ["rosenbrock"]
    func_name_list = ["rosenbrock", "booth", "sphere", "branin", "cross_in_tray", "holder_table"]
    true_noise_sigma = [{"rosenbrock": 0.0, "booth": 0.0, "sphere": 0.0, "branin": 0.0, "cross_in_tray": 0.0, "holder_table": 0.0},
                             {"rosenbrock": 30, "booth": 30, "sphere": 0.5, "branin": 10, "cross_in_tray": 0.01, "holder_table": 0.3}]
    budget_list = [{"rosenbrock": 100, "booth": 100, "sphere": 100, "branin": 100, "cross_in_tray": 800, "holder_table": 700},
                   {"rosenbrock": 300, "booth": 200, "sphere": 150, "branin": 300, "cross_in_tray": 1000, "holder_table": 1000}, ]
    th_list = {"rosenbrock": 100, "booth": 500, "sphere": 20, "branin": 100, "cross_in_tray": -1.5, "holder_table": -3}
    acq_func_name = "Ours"
    L_list = [1, 2, 3, 4, 5]
    color_list = ["r", "g", "b", "cyan", "purple", "orange"]
    batch_size_dict = {"Ours": 1, "MELK": 10, "MILE": 1, "RMILE": 1, "Str": 1, "US": 1}
    rule_name_dict = {"Ours": "Ours", "MELK": "MELK", "MILE": "ConfidenceBound", "RMILE": "ConfidenceBound", "Str": "ConfidenceBound", "US": "ConfidenceBound"}
    n_iteration = 5
    delta = 0.99
    start_time = 0
    resolution = 20
    beta = 1.96
    nu = 0.1
    init_sample_size = 10
    num_iters = 500
    step_size = 1.0
    threshold = 0.95
    sample_size = 10000

    np.random.seed(2)

    save_dir = "result/test_func/change_margin"
    os.makedirs(save_dir, exist_ok=True)

    # Set experimental parameters
    setting_list = {}
    for t, func_name in enumerate(func_name_list):
        for n in range(len(true_noise_sigma)):
            # set test function
            target = TestFunction(function_name=func_name, threshold=th_list[func_name], noise_std=true_noise_sigma[n][func_name], resolution=resolution)
            env = target.env_name
            setting_list[env] = {}
            for L in L_list:
                # set stopping criteria
                stopping_criteria = [
                    ProposedCriterion(delta, L, start_time)
                ]

                params = set_params(acq_func_name, L, delta, None, beta, nu, batch_size_dict[acq_func_name], num_iters, step_size)
                setting_list[env][f"{L}"] = ExpSetting(target, acq_func_name, rule_name_dict[acq_func_name], budget_list[n][func_name], batch_size_dict[acq_func_name], n_iteration, stopping_criteria, params)

    seeds = np.random.randint(0, 1000000, [len(setting_list.keys()), n_iteration])
    for e, env in enumerate(setting_list.keys()):
        for l_key in setting_list[env].keys():
            setting = setting_list[env][l_key]
            for i in range(n_iteration):
                np.random.seed(seeds[e, i])
                test_func = setting.target
                # exlore objective function
                setting_name = f"{save_dir}/{env}_{l_key}_{i}.pkl"
                print(setting_name)
                if os.path.isfile(setting_name):
                    lse = utils.deserialize(setting_name)
                else:
                    # execute LSE
                    lse = TestFunctionLSE(obj_func=test_func, acq_func_name=acq_func_name, rule_name=setting.rule_name, kernel=GPy.kern.RBF(2), stopping_criteria=setting.stopping_criteria, params=setting.params, init_sample_size=init_sample_size, mean_function=test_func.threshold)
                    lse.explore(n_explore=setting.n_explore)
                    utils.serialize(lse, setting_name)

                # calc true class
                setting.f1_score[i] = test_func.evaluate_LSE(lse.history["pred_class"])

                # calc stopping timing
                for s, sc in enumerate(lse.stopping_criteria):
                    setting.stopping_score[s, i] = setting.f1_score[i, sc.stop_timing-1]
                    setting.stopping_time[s, i] = setting.batch_size * sc.stop_timing

        plt.figure(0, [7, 7])
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel("F-score")
        plt.ylim(0.5, 1.01)
        for a, L in enumerate(L_list):
            setting = setting_list[env][f"{L}"]
            plt.xlim(0, 1.01 * setting.budget)
            for s in range(len(stopping_criteria)):
                st = setting.stopping_time[s]
                ss = setting.stopping_score[s]
                plt.scatter(st, ss, c=color_list[a], marker="o", label=f"L={L}", zorder=3*len(L_list)-a, s=130, alpha=st_alpha, edgecolors="k")
        for a, L in enumerate(L_list):
            setting = setting_list[env][f"{L}"]
            for i in range(n_iteration):
                plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[a], zorder=len(L_list) - a, lw=2, alpha=seq_alpha)
                if i == 0:
                    plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[a], label=f"L={L}", zorder=len(L_list)-a, lw=2, alpha=seq_alpha)

        if setting.target.function_name == "sphere" and np.isclose(setting.target.noise_std, 0.0):
            plt.legend(borderaxespad=0, ncol=2, framealpha=1.0, handlelength=0.5, fontsize=24, loc="lower right")
            legend = plt.legend(borderaxespad=0, ncol=2, handlelength=0.5, fontsize=24, loc="lower right")
            legend.set_zorder(4*len(L_list))
        plt.savefig(f"{save_dir}/{env}_f1_and_st_change_margin.pdf", bbox_inches="tight")
