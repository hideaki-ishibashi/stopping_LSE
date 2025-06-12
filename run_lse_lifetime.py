import matplotlib.pyplot as plt
import GPy
from model.stopping_criteria import *
from model.LevelSetEstimation import LifetimeDataLSE
from get_dataset.get_lifetime_data import GetLifetimeData
from utils import utils
from utils.exp_setting import ExpSetting
import os

plt.rcParams["font.size"] = 20
plt.rcParams["figure.autolayout"] = True


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
    plt.figure(0)
    th = 230
    # acq_func_list = ["Ours", "MILE"]
    acq_func_list = ["Ours", "MELK", "MILE", "RMILE", "Str", "US"]
    color_list = ["r", "g", "b", "cyan", "purple", "orange"]
    batch_size_dict = {"Ours": 1, "MELK": 10, "MILE": 1, "RMILE": 1, "Str": 1, "US": 1}
    rule_name_dict = {"Ours": "Ours", "MELK": "MELK", "MILE": "ConfidenceBound", "RMILE": "ConfidenceBound", "Str": "ConfidenceBound", "US": "ConfidenceBound"}
    L = 5
    budget = 1500
    n_iteration = 5
    delta = 0.99
    # data_name_list = ["data3", "data4"]
    data_name_list = ["data4"]
    nu = 0.1
    start_time = 0
    init_sample_size = 30
    beta = 1.96
    n_subsample = 300
    num_iters = 500
    step_size = 1
    sample_size = 10000
    threshold = 0.95

    # save_dir = "result/life_time/supp"
    save_dir = "result/life_time/main"
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(2)

    stopping_criteria = [
        ProposedCriterion(delta, L, start_time),
        F1SamplingCriterion(sample_size=sample_size, threshold=threshold, start_timing=start_time),
        FullyClassifiedCriterion(start_timing=start_time),
    ]

    setting_list = {}
    calc_time = np.zeros((len(acq_func_list), n_iteration))
    f1_list = np.zeros((len(acq_func_list), n_iteration, budget))
    stopping_time_list_our = np.zeros((len(acq_func_list), n_iteration))
    stopping_time_list_fc = np.zeros((len(acq_func_list), n_iteration))
    for data_name in data_name_list:
        target = GetLifetimeData(th, data_name)
        env = target.env_name
        setting_list[env] = {}
        for a, acq_func_name in enumerate(acq_func_list):
            params = set_params(acq_func_name, L, delta, None, beta, nu, batch_size_dict[acq_func_name], num_iters, step_size)
            setting_list[env][acq_func_name] = ExpSetting(target, acq_func_name, rule_name_dict[acq_func_name], budget, batch_size_dict[acq_func_name], n_iteration, stopping_criteria, params)

    seeds = np.random.randint(0, 1000000, [len(setting_list.keys()), n_iteration])
    for e, env in enumerate(setting_list.keys()):
        for a, acq_func_name in enumerate(setting_list[env].keys()):
            setting = setting_list[env][acq_func_name]
            target = setting.target
            for i in range(n_iteration):
                np.random.seed(seeds[e, i])
                setting_name = f"{save_dir}/lifetime_{acq_func_name}_{setting.target.env_name}_{i}.pkl"
                print(setting_name)
                if os.path.isfile(setting_name):
                    lse = utils.deserialize(setting_name)
                else:
                    lse = LifetimeDataLSE(target, acq_func_name=acq_func_name, kernel=GPy.kern.RBF(2), rule_name=setting.rule_name, stopping_criteria=stopping_criteria, params=setting.params, init_sample_size=init_sample_size, n_subsample=n_subsample, mean_function=target.threshold)
                    # lse = LifetimeDataLSE(target, acq_func_name=acq_func_name, kernel=GPy.kern.RBF(2, ARD=True), rule_name=setting.rule_name, stopping_criteria=stopping_criteria, params=setting.params, init_sample_size=init_sample_size, n_subsample=n_subsample, mean_function=target.threshold)
                    lse.explore(n_explore=setting.n_explore)
                    utils.serialize(lse, setting_name)
                setting.set_res_lse(lse)
                print(len(setting.lse_list))

                # calc f1 score
                setting.f1_score[i] = setting.target.evaluate_LSE(lse.history["pred_class"])

                # calc stopping timing
                for s, sc in enumerate(lse.stopping_criteria):
                    setting.stopping_score[s, i] = setting.f1_score[i, sc.stop_timing - 1]
                    setting.stopping_time[s, i] = setting.batch_size * sc.stop_timing
                    print(setting.stopping_score[s, i])
                    print(setting.stopping_time[s, i])

        plt.figure(1, [7, 7])
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel("F-score")
        plt.xlim(0, 1.01*budget)
        plt.ylim(0.5, 1.01)
        for a, acq_func_name in enumerate(acq_func_list):
            setting = setting_list[env][acq_func_name]
            for s in range(len(stopping_criteria)):
                if s == 0:
                    if a == 0:
                        plt.scatter(setting.stopping_time[s], setting.stopping_score[s], c=color_list[a], label=f"Ours", marker="o",
                                    zorder=3*len(acq_func_list) - a, s=130, edgecolors="k")
                elif s == 1:
                    plt.scatter(setting.stopping_time[s], setting.stopping_score[s], c=color_list[a], label=f"FS({acq_func_name})", marker=",",
                                zorder=2 * len(acq_func_list) - a, s=130, edgecolors="k")
        for a, acq_func_name in enumerate(acq_func_list):
            setting = setting_list[env][acq_func_name]
            for i in range(n_iteration):
                if i == 0:
                    plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[a], label=acq_func_name, zorder=len(acq_func_list) - a, lw=2,
                             alpha=0.3)
                else:
                    plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[a], zorder=len(acq_func_list) - a, lw=2,
                             alpha=0.3)
        # if e == 0:
        #     plt.legend(borderaxespad=0, ncol=2, framealpha=1.0, handlelength=0.5, fontsize=24, loc="lower right")
        #     legend = plt.legend(borderaxespad=0, ncol=2, handlelength=0.5, fontsize=24, loc="lower right")
        #     legend.set_zorder(4*len(acq_func_list))
        plt.savefig(f"{save_dir}/lifetime_f1_{env}_FS.pdf", bbox_inches="tight")

        plt.figure(1, [7, 7])
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel("F-score")
        plt.xlim(0, 1.01*budget)
        plt.ylim(0.5, 1.01)
        # plt.xticks([100, 500, 1000, 1500])
        for a, acq_func_name in enumerate(acq_func_list):
            setting = setting_list[env][acq_func_name]
            for s in range(len(stopping_criteria)):
                if s == 0:
                    if a == 0:
                        plt.scatter(setting.stopping_time[s], setting.stopping_score[s], c=color_list[a], label=f"Ours", marker="o",
                                    zorder=3*len(acq_func_list) - a, s=130, edgecolors="k")
                elif s == 2:
                    plt.scatter(setting.stopping_time[s], setting.stopping_score[s], c=color_list[a], label=f"FC({acq_func_name})", marker="^",
                                zorder=2*len(acq_func_list) - a, s=130, edgecolors="k")
        for a, acq_func_name in enumerate(acq_func_list):
            setting = setting_list[env][acq_func_name]
            for i in range(n_iteration):
                if i == 0:
                    plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[a], label=acq_func_name, zorder=len(acq_func_list) - a, lw=2,
                             alpha=0.3)
                else:
                    plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[a], zorder=len(acq_func_list) - a, lw=2,
                             alpha=0.3)

        # if e == 0:
        #     plt.legend(borderaxespad=0, ncol=2, framealpha=1.0, handlelength=0.5, fontsize=24, loc="lower right")
        #     legend = plt.legend(borderaxespad=0, ncol=2, handlelength=0.5, fontsize=24, loc="lower right")
        #     legend.set_zorder(4*len(acq_func_list))
        plt.savefig(f"{save_dir}/lifetime_f1_{env}_FC.pdf", bbox_inches="tight")

        plt.figure(0)
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel("F-score")
        for a, acq_func_name in enumerate(setting_list[env].keys()):
            setting = setting_list[env][acq_func_name]
            if acq_func_name == "MELK":
                for i in range(n_iteration):
                    plt.plot(setting.sample_span, setting.f1_score[i], c=color_list[i])
        plt.savefig(f"{save_dir}/{env}_MELK_f1.pdf", bbox_inches="tight")

        plt.figure(0)
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel(r"$|U_\theta|$")
        for a, acq_func_name in enumerate(setting_list[env].keys()):
            setting = setting_list[env][acq_func_name]
            if acq_func_name == "MELK":
                for i in range(n_iteration):
                    lse = setting.lse_list[i]
                    n_U = np.zeros(len(lse.history["unclassified_set"]))
                    for j in range(len(lse.history["unclassified_set"])):
                        n_U[j] = lse.history["unclassified_set"][j].shape[0]
                    plt.plot(setting.sample_span, n_U, c=color_list[i])
            plt.savefig(f"{save_dir}/{env}_MELK_n_candidates.pdf", bbox_inches="tight")

        plt.figure(0)
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel("Kernel length")
        for a, acq_func_name in enumerate(setting_list[env].keys()):
            setting = setting_list[env][acq_func_name]
            if acq_func_name == "MELK":
                for i in range(n_iteration):
                    lse = setting.lse_list[i]
                    n_length = np.zeros(len(lse.history["length"]))
                    for j in range(len(lse.history["length"])):
                        n_length[j] = lse.history["length"][j]
                    plt.plot(setting.sample_span, n_length, c=color_list[i])
            plt.savefig(f"{save_dir}/{env}_MELK_hyper.pdf", bbox_inches="tight")

        plt.figure(0)
        plt.clf()
        plt.xlabel("Sample size")
        plt.ylabel("Kernel length")
        plt.ylim(15.5, 16.5)
        for a, acq_func_name in enumerate(setting_list[env].keys()):
            setting = setting_list[env][acq_func_name]
            if acq_func_name == "MELK":
                for i in range(n_iteration):
                    lse = setting.lse_list[i]
                    n_length = np.zeros(len(lse.history["length"]))
                    for j in range(len(lse.history["length"])):
                        n_length[j] = lse.history["length"][j]
                    plt.plot(setting.sample_span, n_length, c=color_list[i])
            plt.savefig(f"{save_dir}/{env}_MELK_zoomed_hyper.pdf", bbox_inches="tight")
