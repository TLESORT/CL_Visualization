import os

import matplotlib.pyplot as plt
import wandb
import numpy as np


def generate_python_exp_2_run(dataset, pretrained_on, num_tasks, seed, outlayer, architecture, subset):
    if dataset == "Core10Lifelong" and num_tasks != 1:
        scenario = "Domain"
    else:
        scenario = "Disjoint"

    if "Masked" in outlayer:
        outlayer = outlayer.replace("_Masked", "")
        masked_out = " --masked_out"
    else:
        masked_out = ""

    if subset is not None:
        str_subset = f" --subset {subset}"
    else:
        str_subset = ""

    if architecture is not None:
        str_archi = f" --architecture {architecture}"
    else:
        str_archi = ""

    return f"python main.py  --scenario_name {scenario} --num_tasks {num_tasks} --name_algo baseline" \
           f"  --dataset {dataset} --pretrained_on {pretrained_on} --fast" \
           f" --seed {seed} --OutLayer {outlayer}{masked_out}{str_subset}{str_archi}"


def check_one_config_parameter(config_parameter, value_or_list):
    if isinstance(value_or_list, list):
        parameter_ok = (config_parameter in value_or_list)
    else:
        parameter_ok = (config_parameter == value_or_list)
    return parameter_ok


def select_run(dict_config, dataset, pretrained_on, num_tasks, OutLayer, subset, seed, architecture=None):
    dataset_ok = check_one_config_parameter(dict_config["dataset"], dataset)
    if not dataset_ok: return False

    pretrained_on_ok = check_one_config_parameter(dict_config["pretrained_on"], pretrained_on)
    if not pretrained_on_ok: return False

    num_tasks_ok = check_one_config_parameter(dict_config["num_tasks"], num_tasks)
    if not num_tasks_ok: return False

    OutLayer_ok = check_one_config_parameter(dict_config["OutLayer"], OutLayer)
    if not OutLayer_ok: return False

    subset_ok = check_one_config_parameter(dict_config["subset"], subset)
    if not subset_ok: return False

    seed_ok = check_one_config_parameter(dict_config["seed"], seed)
    if not seed_ok: return False

    if dataset in ["Core50", "Core10Lifelong"]:
        architecture_ok = check_one_config_parameter(dict_config["architecture"], architecture)
    else:
        architecture_ok = True
    return architecture_ok


def select_all_experiments(dataset, pretrained_on, num_tasks, list_subset, list_OutLayer, list_seed, architecture):
    api = wandb.Api()
    # Change oreilly-class/cifar to <entity/project-name>
    runs = api.runs("tlesort/CL_Visualization")
    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files

        dict_config = {k: v for k, v in run.config.items() if not k.startswith('_')}

        if run.state == "finished":
            if select_run(dict_config, dataset, pretrained_on, num_tasks, list_OutLayer, list_subset, seed=list_seed,
                          architecture=architecture):
                summary_list.append(run.summary._json_dict)

                # run.config is the input metrics.  We remove special values that start with _.
                config_list.append(dict_config)

                # run.name is the name of the run.
                name_list.append(run.name)

    # import pandas as pd
    # summary_df = pd.DataFrame.from_records(summary_list)
    # config_df = pd.DataFrame.from_records(config_list)
    # name_df = pd.DataFrame({'name': name_list})
    # all_df = pd.concat([name_df, config_df,summary_df], axis=1)
    #
    # all_df.to_csv("project.csv")
    return summary_list, config_list, name_list


def select_some_experiments(name_list, config_list, str_data_type, dataset, pretrained_on, num_tasks, subset, OutLayer,
                            list_seed, architecture, check_doublon=False):
    api = wandb.Api()

    list_results = []
    for seed in list_seed:
        list_values = []
        id_found = None
        for id_name, config in zip(name_list, config_list):
            if select_run(config, dataset, pretrained_on, num_tasks, OutLayer, subset, seed=seed,
                          architecture=architecture):

                if id_found is not None:
                    # we have two time an experiment with the same config
                    print("doublon !!!!!!!!!!!!!!")
                    print(id_name)
                    print(id_found)
                else:
                    run = api.run(f"tlesort/CL_Visualization/{id_name}")
                    history = run.scan_history()
                    for row in history:
                        # we check if the value is in the dictionnary
                        if str_data_type in row:
                            list_values.append(row[str_data_type])
                    id_found = id_name

        if len(list_values) > 0:
            list_results.append(np.array(list_values))
        else:
            if not check_doublon:
                print(f"No seed {seed}-- {dataset}_pt-{pretrained_on}_{num_tasks}_{OutLayer}_{subset}_{architecture}")
                command2run = generate_python_exp_2_run(dataset, pretrained_on, num_tasks, seed, OutLayer, architecture,
                                                        subset)
                print(f"Command : {command2run} \n")
    return list_results


def plot_experiment(str_data_type, name_list, config_list, dataset, pretrained_on,
                    num_tasks, subset, list_OutLayer, list_seed, architecture, name_extension):
    for OutLayer in list_OutLayer:
        list_results = select_some_experiments(name_list, config_list, str_data_type, dataset, pretrained_on,
                                               num_tasks, subset, OutLayer, list_seed, architecture)

        if len(list_results) > 0:

            array_results = np.array(list_results)
            if array_results.shape[1] == 2:
                # modify the array to make it looks like they are 5 epochs
                new_array = np.ones((array_results.shape[0], 5))
                for i in range(array_results.shape[0]):
                    new_array[i, 0] = array_results[i, 0]
                    new_array[i, 1:] = new_array[i, 1:] * array_results[i, 1]
                array_results = new_array

            if len(list_results) > 1:
                mean = array_results.mean(0)
                std = array_results.std(0)
                plt.plot(range(array_results.shape[1]), mean, label=OutLayer)
                plt.fill_between(range(array_results.shape[1]), mean - std, mean + std, alpha=0.4)
            else:
                plt.plot(range(array_results.shape[1]), array_results[0], label=OutLayer)
        else:
            print(
                f"No results found for {OutLayer} -- {dataset}_pt-{pretrained_on}_{num_tasks}_{subset}_{architecture}")

    xcoords = (np.arange(num_tasks) * 5) + 1

    for i, xc in enumerate(xcoords):
        if i == 0:
            plt.axvline(x=xc, color='red', linewidth=0.5, linestyle='-')
        else:
            plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    save_name = os.path.join(Fig_dir,
                             f"{name_extension}_{dataset}_pt-{pretrained_on}_{num_tasks}_{subset}_{architecture}_{str_data_type}.png")
    plt.ylabel(str_data_type)
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_name)
    plt.clf()
    plt.close()
    print("Great Success")


def generate_one_value(value, std, end_line=False):
    str_line = f"${value:.2f}$\mbox"
    str_line = str_line + "{\scriptsize{$\pm" + f"{std:.2f}$" + "}}"
    if end_line:
        str_line = str_line + "\\\\"
    else:
        str_line = str_line + "& "

    return str_line + " \n"


def generate_one_line(Outlayer, Dataset_label, array_values, nb_lines):
    Outlayer_name = Outlayer.replace("_", "-")
    line = ""
    if nb_lines % 2 == 0:
        line += "\\rowcolor{gray!20}"

    line += f"{Outlayer_name} &\n {Dataset_label} &\n"
    nb_values = array_values.shape[1]

    for i in range(nb_values):

        ind_seed = np.where(array_values[:, i] != -1)[0]
        if len(ind_seed) == 0:
            mean = -1.0
            std = -1.0
        elif len(ind_seed) > 1:
            values = array_values[ind_seed, i] * 100  # convert into pourcentage
            mean = values.mean()
            std = values.std()
        else:
            mean = array_values[ind_seed, i][0] * 100
            std = 0.0

        line = line + generate_one_value(mean, std, end_line=(i == (nb_values - 1)))
    return line + " \n"


def generate_table_subset(name_list, summary_list, config_list, str_data_type, dataset, pretrained_on,
                          num_tasks, list_subset, list_OutLayer, list_seed, architecture):
    str_table = ""
    nb_lines = 0

    for OutLayer in list_OutLayer:
        list_outlayer_results = []
        for subset in list_subset:
            list_items_seed = select_summary_items(name_list, summary_list, config_list, str_data_type, dataset,
                                                   pretrained_on,
                                                   num_tasks, subset, OutLayer, list_seed, architecture)
            if len(list_items_seed) > 0:
                list_outlayer_results.append(np.array(list_items_seed))

        if len(list_outlayer_results) > 0:
            outlayer_array = np.array(list_outlayer_results).reshape(-1, len(list_outlayer_results))

            str_table += generate_one_line(OutLayer, dataset, outlayer_array, nb_lines)
            nb_lines += 1

    str_table += "\n \\hline \n"
    print("##########################################################")
    print("##################   TABLE    ############################")
    print("##########################################################")
    print(str_table)

    textfile = open("table.txt", "a")
    a = textfile.write(str_table)
    textfile.close()


def select_summary_items(name_list, summary_list, config_list, str_data_type, dataset, pretrained_on, num_tasks, subset,
                         OutLayer,
                         list_seed, architecture):
    assert len(name_list) == len(summary_list) == len(config_list)
    list_results_seed = []
    for seed in list_seed:
        id_found = None
        for id_name, summary, config in zip(name_list, summary_list, config_list):
            if select_run(config, dataset, pretrained_on, num_tasks, OutLayer, subset, seed=seed,
                          architecture=architecture):
                if id_found is not None:
                    # we have two time an experiment with the same config
                    print("doublon !!!!!!!!!!!!!!")
                    print(id_name)
                    print(id_found)
                else:
                    if str_data_type in summary:
                        list_results_seed.append(summary[str_data_type])
                        id_found = id_name

        if id_found is None:
            list_results_seed.append(-1)
            print(f"No seed {seed}-- {dataset}_pt-{pretrained_on}_{num_tasks}_{OutLayer}_{subset}_{architecture}")
            command2run = generate_python_exp_2_run(dataset, pretrained_on, num_tasks, seed, OutLayer, architecture,
                                                    subset)
            print(f"Command : {command2run} \n")
    return list_results_seed


def plot_experiment_subset(str_data_type, name_list, config_list, dataset, pretrained_on,
                           num_tasks, list_subset, list_OutLayer, list_seed, architecture):
    for subset in list_subset:
        plot_experiment(str_data_type, name_list, config_list, dataset, pretrained_on,
                        num_tasks, subset, list_OutLayer, list_seed, architecture)

    for OutLayer in list_OutLayer:
        for subset in list_subset:
            list_results = select_some_experiments(name_list, config_list, str_data_type, dataset, pretrained_on,
                                                   num_tasks, subset, OutLayer, list_seed, architecture)
            if len(list_results) > 0:

                array_results = np.array(list_results)

                if len(list_results) > 1:
                    mean = array_results.mean(0)
                    std = array_results.std(0)
                    plt.plot(range(array_results.shape[1]), mean, label=f"Subset-{subset}")
                    plt.fill_between(range(array_results.shape[1]), mean - std, mean + std, alpha=0.4)
                else:
                    plt.plot(range(array_results.shape[1]), array_results[0], label=f"Subset-{subset}")
            else:
                print(
                    f"No results found for {OutLayer} -- {dataset}_pt-{pretrained_on}_{num_tasks}_{subset}_{architecture}")

        save_name = os.path.join(Fig_dir,
                                 f"{dataset}_pt-{pretrained_on}_{num_tasks}_Subsets_{OutLayer}_{architecture}_{str_data_type}.png")
        plt.ylabel(str_data_type)
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(save_name)
        plt.clf()
        plt.close()
        print("Great Success")


pretrained_on = "ImageNet"
list_subset = [100, 200, 500, 1000, 10000, None]
Fig_dir = "./Archives/Figures/wandb"
architecture = "resnet"
if not os.path.exists(Fig_dir):
    os.makedirs(Fig_dir)
list_OutLayer = ["Linear", "Linear_no_bias", "CosLayer", "SLDA", "MeanLayer", "MedianLayer", "KNN",
                 "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked",
                 'MIMO_Linear', "MIMO_CosLayer", "MIMO_Linear_no_bias",
                 'MIMO_Linear_Masked', "MIMO_CosLayer_Masked", "MIMO_Linear_no_bias_Masked"]
list_OutLayer = ["Linear", "Linear_no_bias", "CosLayer", "SLDA", "MeanLayer", "MedianLayer",
                 "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked",
                 'MIMO_Linear_Masked', "MIMO_CosLayer_Masked", "MIMO_Linear_no_bias_Masked"]
list_seed = [0, 1]

list_dataset = ["CIFAR10", "CIFAR100", "CIFAR10", "CIFAR100", "Core50", "Core10Lifelong"]
list_architecture = [None, None, None, None, "resnet", "resnet"]
list_pretrained_on = ["CIFAR100", "CIFAR10", "CIFAR10", "CIFAR100", "ImageNet", "ImageNet"]
list_num_tasks = [5, 10, 5, 10, 10, 8]

check_for_doublon = False
if check_for_doublon:
    for dataset, pretrained_on, num_tasks, architecture in zip(list_dataset, list_pretrained_on, list_num_tasks,
                                                               list_architecture):
        print(dataset)

        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer,
                                                                      list_seed, architecture=architecture)

        for OutLayer in list_OutLayer:
            for subset in list_subset:
                list_results = select_some_experiments(name_list, config_list, "test accuracy", dataset, pretrained_on,
                                                       num_tasks, subset, OutLayer, list_seed, architecture,
                                                       check_doublon=True)

if False:
    ####################################################################################
    # Masked Experiments
    ####################################################################################

    list_dataset = ["CIFAR10", "CIFAR100", "Core50", "Core10Lifelong"]
    list_architecture = [None, None, "resnet", "resnet"]
    list_pretrained_on = ["CIFAR100", "CIFAR10", "ImageNet", "ImageNet"]
    list_num_tasks = [5, 10, 10, 8]

    list_OutLayer_masked_exp = ["Linear", "Linear_no_bias", "CosLayer",
                                "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked"]

    for dataset, pretrained_on, num_tasks, architecture in zip(list_dataset, list_pretrained_on, list_num_tasks,
                                                               list_architecture):
        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_masked_exp,
                                                                      list_seed, architecture=architecture)

        nb_exps = len(summary_list)

        plot_experiment(str_data_type="test accuracy", name_list=name_list, config_list=config_list, dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=list_OutLayer_masked_exp,
                        list_seed=list_seed,
                        architecture=architecture, name_extension="Masked_Exp")

    ####################################################################################
    # Continual Experiments
    ####################################################################################

    list_dataset = ["CIFAR100"]
    list_pretrained_on = ["CIFAR10"]
    list_num_tasks = [10]

    list_dataset = ["CIFAR10", "CIFAR100", "Core50", "Core10Lifelong"]
    list_pretrained_on = ["CIFAR100", "CIFAR10", "ImageNet", "ImageNet"]
    list_num_tasks = [5, 10, 10, 8]

    list_dataset = ["CIFAR100"]
    list_pretrained_on = ["CIFAR10"]
    list_num_tasks = [10]

    for dataset, pretrained_on, num_tasks in zip(list_dataset, list_pretrained_on, list_num_tasks):
        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer,
                                                                      list_seed, architecture=None)

        nb_exps = len(summary_list)

        plot_experiment(str_data_type="test accuracy", name_list=name_list, config_list=config_list, dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=list_OutLayer, list_seed=list_seed,
                        architecture=None)

####################################################################################
# Subset Experiments
####################################################################################
list_subset = [100, 200, 500, 1000, 10000, None]

list_dataset = ["Core50", "Core10Lifelong"]
list_pretrained_on = ["ImageNet", "ImageNet"]
list_num_tasks = [1, 1]
architecture = "resnet"

# reset file
textfile = open("table.txt", "w")
a = textfile.write("")
textfile.close()

for dataset, pretrained_on, num_tasks in zip(list_dataset, list_pretrained_on, list_num_tasks):
    summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                  list_OutLayer,
                                                                  list_seed, architecture=architecture)

    nb_exps = len(summary_list)

    # plot_experiment_subset(str_data_type="test accuracy", name_list=name_list, config_list=config_list,
    #                        dataset=dataset,
    #                        pretrained_on=pretrained_on,
    #                        num_tasks=num_tasks, list_subset=list_subset, list_OutLayer=list_OutLayer,
    #                        list_seed=list_seed, architecture=None)
    generate_table_subset(name_list, summary_list, config_list, "test accuracy", dataset, pretrained_on,
                          num_tasks, list_subset, list_OutLayer, list_seed, architecture=architecture)
