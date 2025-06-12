import matplotlib.pyplot as plt
import GPy
import matplotlib.patheffects as mpe
from model.stopping_criteria import *
from model.LevelSetEstimation import TestFunctionLSE
from get_dataset.test_func import TestFunction
import os
from utils import utils
from utils.exp_setting import ExpSetting

plt.rcParams["font.size"] = 24
plt.rcParams["figure.autolayout"] = True
lw = 2
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
    true_noise_sigma = {"rosenbrock": [0.0, 10.0], "sphere": [0.0, 0.5]}
    func_name_list = ["rosenbrock"]
    # func_name_list = ["rosenbrock", "sphere"]
    budget_list = {"rosenbrock": [100, 500], "sphere": [100, 300]}
    th_list = {"rosenbrock": 100, "sphere": 20}
    acq_func_name = "Ours"
    margin = None
    jitter_list = {"rosenbrock": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], "sphere": [-0.125, -0.075, -0.025, 0.025, 0.075, 0.125]}
    L = 3.0
    color_list = ["r", "g", "b", "cyan", "purple", "orange"]
    batch_size_dict = {"Ours": 1, "MELK": 10, "MILE": 1, "RMILE": 1, "Str": 1, "US": 1}
    rule_name_dict = {"Ours": "Ours", "MELK": "MELK", "MILE": "ConfidenceBound", "RMILE": "ConfidenceBound", "Str": "ConfidenceBound", "US": "ConfidenceBound"}
    n_iteration = 5
    batch_size = 1
    delta = 0.99
    start_time = 20
    resolutions = [20, 50, 100, 200, 300, 400]
    beta = 1.96
    nu = 0.1
    num_iters = 1000
    step_size = 1.0
    init_sample_size = 10
    isReclassified = True

    np.random.seed(2)

    save_dir = "result/test_func/change_n_candidates"
    os.makedirs(save_dir, exist_ok=True)

    # Set experimental parameters
    setting_list = {}
    for t, func_name in enumerate(func_name_list):
        for n in range(len(true_noise_sigma[func_name])):
            # set stopping criteria
            stopping_criteria = [
                ProposedCriterion(delta, L, start_time),
                FullyClassifiedCriterion(start_timing=start_time),
            ]

            # set test function
            env = f"{func_name}_{th_list[func_name]}_{true_noise_sigma[func_name][n]}"
            setting_list[env] = {}
            for m, resolution in enumerate(resolutions):
                target = TestFunction(function_name=func_name, threshold=th_list[func_name],
                                      noise_std=true_noise_sigma[func_name][n], resolution=resolution)
                params = set_params(acq_func_name, L, delta, None, beta, nu, batch_size_dict[acq_func_name], num_iters, step_size)
                setting_list[env][f"{resolution}"] = ExpSetting(target, acq_func_name, rule_name_dict[acq_func_name], budget_list[func_name][n], batch_size_dict[acq_func_name], n_iteration, stopping_criteria, params)

    seeds = np.random.randint(0, 1000000, [len(func_name_list), len(true_noise_sigma["rosenbrock"]), n_iteration])
    for t, func_name in enumerate(func_name_list):
        for n in range(len(true_noise_sigma[func_name])):
            env = f"{func_name}_{th_list[func_name]}_{true_noise_sigma[func_name][n]}"
            for m, resolution in enumerate(setting_list[env].keys()):
                setting = setting_list[env][f"{resolution}"]
                for i in range(n_iteration):
                    np.random.seed(seeds[t, n, i])
                    test_func = setting.target
                    # exlore objective function
                    setting_name = f"{save_dir}/{env}_{resolution}_{i}.pkl"
                    stopping_criteria = setting_list[env][f"{resolution}"].stopping_criteria
                    print(setting_name)
                    if os.path.isfile(setting_name):
                        lse = utils.deserialize(setting_name)
                    else:
                        # execute LSE
                        lse = TestFunctionLSE(obj_func=test_func, acq_func_name=acq_func_name,
                                              rule_name=setting.rule_name, kernel=GPy.kern.RBF(2),
                                              stopping_criteria=stopping_criteria, params=setting.params,
                                              init_sample_size=init_sample_size, mean_function=test_func.threshold)
                        lse.explore(n_explore=setting.n_explore)
                        utils.serialize(lse, setting_name)
                    # calc true class
                    setting.f1_score[i] = test_func.evaluate_LSE(lse.history["pred_class"])

                    # calc stopping timing
                    for s, sc in enumerate(lse.stopping_criteria):
                        setting.stopping_score[s, i] = setting.f1_score[i, sc.stop_timing-1]
                        setting.stopping_time[s, i] = setting.batch_size * sc.stop_timing

    for t, func_name in enumerate(func_name_list):
        plt.figure(0, [7, 7])
        for n in range(len(true_noise_sigma[func_name])):
            plt.clf()
            plt.ylim(0.0, 1.01)
            env = f"{func_name}_{th_list[func_name]}_{true_noise_sigma[func_name][n]}"
            plt.xlabel("Number of candidate points")
            plt.ylabel("F-score")
            for m, resolution in enumerate(resolutions):
                print(setting_list)
                setting = setting_list[env][f"{resolution}"]
                stopping_criteria = setting_list[env][f"{resolution}"].stopping_criteria
                ss = setting.stopping_score[0]
                plt.scatter(resolution ** 2 * np.ones(ss.shape[0]), ss,
                            c=color_list[m], marker="o", zorder=3 * len(resolutions) - m,
                            s=130, alpha=st_alpha, edgecolors="k")
                outline = mpe.withStroke(linewidth=4, foreground=color_list[m], capstyle="projecting")
                plt.errorbar(resolution**2, ss.mean(), yerr=ss.std(),
                             c="white", zorder=4 * len(resolutions) - m, capsize=5, lw=2, path_effects=[outline])
            # if func_name == "rosenbrock":
            #     plt.legend(handletextpad=0, fontsize=24, columnspacing=1)
            plt.savefig(f"{save_dir}/change_candidate_{env}_f1.pdf", bbox_inches="tight")

    for t, func_name in enumerate(func_name_list):
        plt.figure(0, [7, 7])
        for n in range(len(true_noise_sigma[func_name])):
            plt.clf()
            plt.ylim(0.0, budget_list[func_name][n])
            env = f"{func_name}_{th_list[func_name]}_{true_noise_sigma[func_name][n]}"
            plt.xlabel("The number of candidate points")
            plt.ylabel("Stopping time")
            for m, resolution in enumerate(resolutions):
                setting = setting_list[env][f"{resolution}"]
                stopping_criteria = setting_list[env][f"{resolution}"].stopping_criteria
                st = setting.stopping_time[0]
                plt.scatter(resolution ** 2 * np.ones(st.shape[0]), st,
                            c=color_list[m], marker="o", zorder=3 * len(resolutions) - m,
                            s=130, alpha=st_alpha, edgecolors="k")
                outline = mpe.withStroke(linewidth=4, foreground=color_list[m], capstyle="projecting")
                plt.errorbar(resolution**2, st.mean(), yerr=st.std(),
                             c="white", zorder=4 * len(resolutions) - m, capsize=5, lw=2, path_effects=[outline])
            plt.savefig(f"{save_dir}/change_candidate_{env}_st.pdf", bbox_inches="tight")
    for t, func_name in enumerate(func_name_list):
        for n in range(len(true_noise_sigma[func_name])):
            env = f"{func_name}_{th_list[func_name]}_{true_noise_sigma[func_name][n]}"
            for m, resolution in enumerate(resolutions):
                setting = setting_list[env][f"{resolution}"]
                stopping_criteria = setting_list[env][f"{resolution}"].stopping_criteria
                ss = setting.stopping_score[0]
                st = setting.stopping_time[0]
                print(f"{resolution**2}: {ss.mean()}+-{ss.std()} | {st.mean()}+-{st.std()}")
