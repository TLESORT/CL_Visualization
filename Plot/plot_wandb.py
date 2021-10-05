import os

import matplotlib.pyplot as plt
import wandb
import numpy as np
from itertools import cycle

import sys
sys.path.append("..")
from utils_wandb import select_run

def generate_python_exp_2_run(dataset, pretrained_on, num_tasks, seed, outlayer, architecture, subset, lr):
    if dataset in ["Core10Lifelong", "Core10Mix", "CIFAR100Lifelong"] and num_tasks != 1:
        scenario = "Domain"
    else:
        scenario = "Disjoint"

    if "_Masked" in outlayer:
        outlayer = outlayer.replace("_Masked", "")
        masked_out = " --masked_out single"
    elif "_GMasked" in outlayer:
        outlayer = outlayer.replace("_GMasked", "")
        masked_out = " --masked_out group"
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
           f"  --dataset {dataset} --pretrained_on {pretrained_on} --fast --lr {lr}" \
           f" --seed {seed} --OutLayer {outlayer}{masked_out}{str_subset}{str_archi}"


def select_all_experiments(dataset, pretrained_on, num_tasks, list_subset, list_OutLayer, list_seed, lr, architecture):
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
            if select_run(dict_config=dict_config, name_algo="baseline", dataset=dataset, pretrained_on=pretrained_on,
                          num_tasks=num_tasks, OutLayer=list_OutLayer, subset=list_subset, seed=list_seed,
                          lr=lr, architecture=architecture, finetuning=False, test_label=False):
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
                            list_seed, lr, architecture, check_doublon=False):
    api = wandb.Api()

    if dataset in ["Core10Lifelong", "Core10Mix", "CIFAR100Lifelong"]  and num_tasks>1:
        name_scenario ="Domain"
    else:
        name_scenario ="Disjoint"


    list_results = []
    for seed in list_seed:
        list_values = []
        id_found = None
        for id_name, config in zip(name_list, config_list):

            if select_run(dict_config=config, name_algo="baseline", dataset=dataset,
                          pretrained_on=pretrained_on,
                          num_tasks=num_tasks, OutLayer=OutLayer, subset=subset, seed=seed,
                          lr=lr, architecture=architecture, finetuning=False, test_label=False):

                if id_found is not None:
                    # we have two time an experiment with the same config
                    print("select_some_experiments")
                    print("doublon !!!!!!!!!!!!!!")
                    print(config)
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
                # Core10Lifelong-Disjoint-1-tasks-pretrained_on_ImageNet-SingleH-seed-1-CosLayer
                print(f"No seed {seed} subset {subset} archi {architecture} \n"
                      f" name: {dataset}-{name_scenario}-{num_tasks}-tasks-pretrained_on_{pretrained_on}-SingleH-seed-{seed}-{OutLayer}")
                command2run = generate_python_exp_2_run(dataset, pretrained_on, num_tasks, seed, OutLayer, architecture,
                                                        subset, lr)
                print(f"Command : {command2run} \n")
    return list_results

def export_legend(ax, filename, three_columns=False):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    if three_columns:
        ncol=3
    else:
        ncol = 5
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=True, loc='lower center', ncol=ncol,)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def select_style(outlayer):

    if "GMasked" in outlayer:
        style = ':'
    elif "Masked" in outlayer:
        style = '--'
    elif outlayer in ["KNN", "SLDA", "MeanLayer", "MedianLayer"]:
        style = '-.'
    else:
        style = '-'
    return style

# def select_color(outlayer):
#
#     if "Linear_no_bias" in outlayer:
#         color = 'blue'
#     elif "Linear" in outlayer:
#         color = 'cyan'
#     elif "CosLayer" in outlayer:
#         color = 'green'
#     else:
#         style = '-'
#     return style


def plot_experiment(str_data_type, name_list, config_list, dataset, pretrained_on,
                    num_tasks, subset, list_OutLayer, list_seed, lr, architecture, name_extension, legend=False):

    NB_EPOCHS = 5
    #style_c = cycle(['-', '--', ':', '-.'])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if legend:
        list_seed=[list_seed[0]] # we just take on seed to just plot the legend

    for OutLayer in list_OutLayer:

        if ("_GMasked" in OutLayer) and (dataset in ["Core10Lifelong", "Core10Mix", "CIFAR100Lifelong"]):
            # Gmasked has no interest here
            continue

        if OutLayer in ["KNN", "SLDA", "MeanLayer", "MedianLayer"]:
            lr_value = 0.002 # default lr for layer without lr
        elif "Linear" in OutLayer:
            lr_value = 0.01
        else:
            lr_value = lr

        list_results = select_some_experiments(name_list, config_list, str_data_type, dataset, pretrained_on,
                                               num_tasks, subset, OutLayer, list_seed, lr_value, architecture)


        if len(list_results) > 0:

            array_results = np.array(list_results)

            print(f"Layer {OutLayer}")
            print(array_results.shape)

            if array_results.shape[1] == 2: # THIS IS FOR KNN - MEANLAYER - MEDIAN LAYER - SLDA
                # modify the array to make it looks like they are 5 epochs
                new_array = np.ones((array_results.shape[0], NB_EPOCHS))
                for i in range(array_results.shape[0]):
                    new_array[i, 0] = array_results[i, 0]
                    new_array[i, 1:] = new_array[i, 1:] * array_results[i, 1]
                array_results = new_array

            style = select_style(OutLayer)
            if len(list_results) > 1:
                mean = array_results.mean(0)
                std = array_results.std(0)
                ax.plot(range(array_results.shape[1]), mean, label=OutLayer, linestyle=style)
                ax.fill_between(range(array_results.shape[1]), mean - std, mean + std, alpha=0.4)
            else:
                ax.plot(range(array_results.shape[1]), array_results[0], label=OutLayer, linestyle=style)
        else:
            print(
                f"No results found for {OutLayer} -- {dataset}_pt-{pretrained_on}_{num_tasks}_{subset}_{architecture}")



    if legend:
        legend_name = os.path.join(Fig_dir,
                                 f"legend_{name_extension}_{dataset}_pt-{pretrained_on}_{num_tasks}_{subset}_{architecture}_{str_data_type}.png")
        export_legend(ax, legend_name, dataset=="Core10Mix")
    else:
        xcoords = (np.arange(num_tasks) * NB_EPOCHS) #+ 1

        for i, xc in enumerate(xcoords):
            if i == 0:
                plt.axvline(x=xc, color='red', linewidth=0.5, linestyle='-')
            else:
                plt.axvline(x=xc, color='#929591', linewidth=0.1, linestyle='-.')

        save_name = os.path.join(Fig_dir,
                                 f"{name_extension}_{dataset}_pt-{pretrained_on}_{num_tasks}_{subset}_{architecture}_{str_data_type}.png")
        plt.ylabel(str_data_type)
        plt.xlabel('Epochs')
        plt.ylim(0, 1.0)

        #plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05),ncol=5, bbox_transform=plt.gcf().transFigure)
        print(f"Save : {save_name}")
        plt.savefig(save_name, bbox_inches='tight')
        plt.savefig(save_name.replace(".png",".pdf"), bbox_inches='tight')
        plt.clf()
        plt.close()
        print("Great Success")


def generate_one_value(value, std, end_line=False, comment=None):
    str_line = ""
    if value == -1:
        str_line = str_line + " - "
    else:
        str_line += f"${value:.2f}$\mbox"
        str_line += "{\scriptsize{$\pm" + f"{std:.2f}$" + "}}"
    if end_line:
        str_line += "\\\\"
    else:
        str_line += "& "

    if comment is not None:
        str_line += f"% {comment} "

    return str_line + " \n"


def generate_one_line(list_value_name, array_values, nb_lines, list_comment):

    line = ""
    if nb_lines % 2 == 1:
        line += "\\rowcolor{gray!20}"

    for name in list_value_name:
        if name is None:
            line += f"  -   & \n"
        else:
            name = name.replace("_", "-")
            line += f"{name} & \n"

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

        line = line + generate_one_value(mean, std, end_line=(i == (nb_values - 1)), comment=list_comment[i])
    return line + " \n"


def generate_table_subset(name_list, summary_list, config_list, str_data_type, dataset, pretrained_on,
                          num_tasks, list_subset, list_OutLayer, list_seed, lr, architecture):
    str_table = f"% ##############  {lr} ############## \n"
    nb_lines = 0

    for OutLayer in list_OutLayer:
        #list_outlayer_results = []
        outlayer_array = np.zeros((len(list_seed), 0))

        for subset in list_subset:
            if OutLayer in ["KNN", "SLDA", "MeanLayer", "MedianLayer"]:
                lr_value = 0.002  # default lr for layer without lr
            elif "Linear" in OutLayer:
                lr_value = 0.01
            else:
                lr_value = lr
            list_items_seed = select_summary_items(name_list, summary_list, config_list, str_data_type, dataset,
                                                   pretrained_on,
                                                   num_tasks, subset, OutLayer, list_seed, lr_value, architecture)
            if len(list_items_seed) > 0:
                outlayer_array = np.concatenate([outlayer_array, np.array(list_items_seed).reshape(len(list_seed), 1)], axis=1)
                #list_outlayer_results.append(np.array(list_items_seed))

        #if len(list_outlayer_results) > 0:
        if outlayer_array.shape[0] > 0:

            str_table += generate_one_line(list_value_name = [OutLayer, dataset],
                                           array_values=outlayer_array,
                                           nb_lines=nb_lines,
                                           list_comment=list_subset)
            nb_lines += 1

    str_table += "\n \\hline \n"
    # print("##########################################################")
    # print("##################   TABLE    ############################")
    # print("##########################################################")
    # print(str_table)

    textfile = open("table.txt", "a")
    a = textfile.write(str_table)
    textfile.close()

def generate_table_preliminary(name_list, summary_list, config_list, str_data_type, dataset, pretrained_on,
                          num_tasks, subset, list_OutLayer, list_seed, list_lr, architecture):
    str_table = f""
    nb_lines = 0

    print("We start generating the table!")

    for lr in list_lr:
        list_outlayer_results = []
        outlayer_array = np.ones((len(list_seed), len(list_OutLayer))) * -1
        for o, OutLayer in enumerate(list_OutLayer):
            list_items_seed = select_summary_items(name_list, summary_list, config_list, str_data_type, dataset,
                                                   pretrained_on,
                                                   num_tasks, subset, OutLayer, list_seed, lr, architecture)
            if len(list_items_seed) > 0:
                outlayer_array[:len(list_items_seed), o] = np.array(list_items_seed)
                #list_outlayer_results.append(np.array(list_items_seed))

        #if len(list_outlayer_results) > 0:
        #outlayer_array = np.array(list_outlayer_results).reshape(-1, len(list_outlayer_results))
        print(outlayer_array)

        if lr == 0.002:
            lr_name = 'n/a'
        else:
            lr_name = str(lr)

        str_table += generate_one_line(list_value_name=[dataset, lr_name, architecture, pretrained_on],
                                       array_values=outlayer_array, nb_lines=nb_lines, list_comment=list_OutLayer)
        nb_lines += 1
    # else: # For approaches with learning rate
    #     print("we will think about it later, it is for KNN / MeanLayer ...")


    str_table += "\n \\hline \n"
    textfile = open("table.txt", "a")
    a = textfile.write(str_table)
    textfile.close()

def select_summary_items(name_list, summary_list, config_list, str_data_type, dataset, pretrained_on, num_tasks, subset,
                         OutLayer,
                         list_seed, lr, architecture):
    assert len(name_list) == len(summary_list) == len(config_list)
    list_results_seed = []
    for seed in list_seed:
        id_found = None
        for id_name, summary, config in zip(name_list, summary_list, config_list):

            if select_run(dict_config=config, name_algo="baseline", dataset=dataset,
                          pretrained_on=pretrained_on,
                          num_tasks=num_tasks, OutLayer=OutLayer, subset=subset, seed=seed,
                          lr=lr, architecture=architecture, finetuning=False, test_label=False):
                if id_found is not None:
                    # we have two time an experiment with the same config
                    print("select_summary_items")
                    print("doublon !!!!!!!!!!!!!!")
                    print(config)
                    print(id_name)
                    print(id_found)
                else:
                    if str_data_type in summary:
                        list_results_seed.append(summary[str_data_type])
                        id_found = id_name

        if id_found is None:
            list_results_seed.append(-1)
            print(f"No seed {seed} subset {subset} archi {architecture} \n"
                  f" name: {dataset}-Disjoint-{num_tasks}-tasks-pretrained_on_{pretrained_on}-SingleH-seed-{seed}-{OutLayer}")
            command2run = generate_python_exp_2_run(dataset, pretrained_on, num_tasks, seed, OutLayer, architecture,
                                                    subset, lr)
            print(f"Command : {command2run} \n")
    return list_results_seed


def plot_experiment_subset(str_data_type, name_list, config_list, dataset, pretrained_on,
                           num_tasks, list_subset, list_OutLayer, list_seed, lr, architecture):
    for subset in list_subset:
        plot_experiment(str_data_type, name_list, config_list, dataset, pretrained_on,
                        num_tasks, subset, list_OutLayer, list_seed, lr,  architecture, legend=False)
        plot_experiment(str_data_type, name_list, config_list, dataset, pretrained_on,
                        num_tasks, subset, list_OutLayer, list_seed, lr,  architecture, legend=True)

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
        plt.ylim(0, 1.0)
        plt.legend(loc='top center')
        print(f"Save : {save_name}")
        plt.savefig(save_name, bbox_inches='tight')
        plt.savefig(save_name.replace(".png",".pdf"), bbox_inches='tight')
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
                 "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked"]


list_dataset = ["CIFAR10", "CIFAR100", "CIFAR10", "CIFAR100", "Core50", "Core10Lifelong"]
list_architecture = [None, None, None, None, "resnet", "resnet"]
list_pretrained_on = ["CIFAR100", "CIFAR10", "CIFAR10", "CIFAR100", "ImageNet", "ImageNet"]
list_num_tasks = [5, 10, 5, 10, 10, 8]
list_seed = [0, 1]
lr = 0.002

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
                                                       num_tasks, subset, OutLayer, list_seed, lr, architecture,
                                                       check_doublon=True)

if False:
    ####################################################################################
    # Préselection Experiments
    ####################################################################################
    list_subset = [None]
    list_seed = [0, 1, 2, 3, 4, 5, 6, 7]

    list_dataset = ["CIFAR10", "CIFAR10", "CIFAR100", "CIFAR100"]
    list_architecture = ["resnet", "resnet", "resnet", "resnet"]
    list_pretrained_on = ["CIFAR10", "CIFAR100", "CIFAR10", "CIFAR100"]

    list_dataset = ["Core50", "Core50", "Core50", "Core10Lifelong","Core10Lifelong","Core10Lifelong"]
    list_architecture = ["resnet", "vgg", "googlenet","resnet", "vgg",  "googlenet"]
    list_pretrained_on = ["ImageNet", "ImageNet", "ImageNet", "ImageNet", "ImageNet", "ImageNet"]
    #
    # list_dataset = ["CIFAR10", "CIFAR10", "CIFAR100", "CIFAR100", "Core50", "Core10Lifelong"]
    # list_architecture = ["resnet", "resnet", "resnet", "resnet","resnet","resnet"]
    # list_pretrained_on = ["CIFAR10", "CIFAR100", "CIFAR10", "CIFAR100", "ImageNet", "ImageNet"]

    list_lr = [0.1, 0.01, 0.001]
    #list_lr = [0.1]

    list_OutLayer = ["Linear", "Linear_no_bias", "CosLayer", "WeightNorm", "OriginalWeightNorm"] # ,"Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked"]
    #list_OutLayer = [] # ,"Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked"]
    list_OutLayer_no_LR = ["KNN", "SLDA", "MeanLayer"] #, "MedianLayer"]

    # reset file
    textfile = open("table.txt", "w")
    a = textfile.write("")
    textfile.close()

    for dataset, pretrained_on, architecture in zip(list_dataset, list_pretrained_on, list_architecture):
        print(dataset)
        full_summary_list, full_config_list, full_name_list = [], [], []
        for lr in list_lr:
            summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks=1, list_subset=list_subset,
                                                                          list_OutLayer=list_OutLayer,
                                                                          list_seed=list_seed, lr=lr, architecture=architecture)
            full_summary_list=full_summary_list+summary_list
            full_config_list=full_config_list+config_list
            full_name_list=full_name_list+name_list


        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks=1,
                                                                      list_subset=list_subset,
                                                                      list_OutLayer=list_OutLayer_no_LR,
                                                                      list_seed=list_seed, lr=0.002, architecture=architecture)
        full_summary_list = full_summary_list + summary_list
        full_config_list = full_config_list + config_list
        full_name_list = full_name_list + name_list
        tot_list_lr = list_lr + [0.002] # default value for list_OutLayer_no_LR layers
        tot_list_OutLayer = list_OutLayer + list_OutLayer_no_LR

        generate_table_preliminary(full_name_list, full_summary_list, full_config_list, "test accuracy", dataset, pretrained_on,
                                   num_tasks=1, subset=None, list_OutLayer=tot_list_OutLayer, list_seed=list_seed, list_lr=tot_list_lr, architecture=architecture)



####################################################################################
# Masked Experiments
####################################################################################
if True:

    list_dataset = ["CIFAR10", "CIFAR100", "Core50", "Core10Lifelong"]
    list_pretrained_on = ["CIFAR100", "CIFAR10", "ImageNet", "ImageNet"]
    list_architecture = [None, None, "resnet", "resnet"]
    list_num_tasks = [5, 10, 10, 8]


    list_dataset = ["CIFAR10", "CIFAR100Lifelong","Core50", "Core10Lifelong", "Core10Mix", "CUB200"]
    list_pretrained_on = ["CIFAR100", "CIFAR100", "ImageNet", "ImageNet", "ImageNet", "ImageNet"]
    list_architecture = [None, None, "resnet", "resnet", "resnet", "resnet"]
    list_num_tasks = [5, 5, 10, 8, 50, 10]

    #
    # list_dataset = ["CUB200"]
    # list_pretrained_on = ["ImageNet"]
    # list_architecture = ["resnet"]
    # list_num_tasks = [10]

    # list_dataset = ["CIFAR10"]
    # list_pretrained_on = ["CIFAR100"]
    # list_architecture = [None]
    # list_num_tasks = [5]



    # list_dataset = ["Core50", "Core10Lifelong"]
    # list_pretrained_on = ["ImageNet", "ImageNet"]
    # list_architecture = ["vgg", "vgg"]
    # list_num_tasks = [10, 8]


    list_subset = [None]
    list_seed = [0, 1, 2, 3, 4, 5, 6, 7]
    lr=0.1
    list_OutLayer_masked_exp_01 = ["CosLayer", "WeightNorm", "OriginalWeightNorm",
                                "CosLayer_Masked", "WeightNorm_Masked", "OriginalWeightNorm_Masked",
                                "CosLayer_GMasked", "WeightNorm_GMasked", "OriginalWeightNorm_GMasked" ]

    list_OutLayer_masked_exp_001 = ["Linear", "Linear_no_bias",
                                   "Linear_Masked", "Linear_no_bias_Masked",
                                   "Linear_GMasked", "Linear_no_bias_GMasked"]

    list_OutLayer_masked_exp = ["Linear", "Linear_no_bias", "CosLayer", "WeightNorm", "OriginalWeightNorm",
                                "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked", "WeightNorm_Masked", "OriginalWeightNorm_Masked",
                                "Linear_GMasked", "Linear_no_bias_GMasked", "CosLayer_GMasked", "WeightNorm_GMasked", "OriginalWeightNorm_GMasked" ]


    for dataset, pretrained_on, num_tasks, architecture in zip(list_dataset, list_pretrained_on, list_num_tasks,
                                                               list_architecture):
        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_masked_exp_01,
                                                                      list_seed, lr=0.1, architecture=architecture)

        summary_list_2, config_list_2, name_list_2 = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_masked_exp_001,
                                                                      list_seed, lr=0.01, architecture=architecture)

        summary_list = summary_list + summary_list_2
        config_list = config_list + config_list_2
        name_list = name_list + name_list_2
        nb_exps = len(summary_list)

        plot_experiment(str_data_type="test accuracy", name_list=name_list, config_list=config_list, dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=list_OutLayer_masked_exp,
                        list_seed=list_seed, lr=lr,
                        architecture=architecture, name_extension="Masked_Exp")

        plot_experiment(str_data_type="test accuracy", name_list=name_list, config_list=config_list, dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=list_OutLayer_masked_exp,
                        list_seed=list_seed, lr=lr,
                        architecture=architecture, name_extension="Masked_Exp", legend=True)

####################################################################################
# Comparison Gradient vs prototypes Experiments
####################################################################################
if False:

    list_OutLayer_grad_exp = ["CosLayer", "WeightNorm",
                                "CosLayer_Masked", "WeightNorm_Masked",
                                "CosLayer_GMasked", "WeightNorm_GMasked"]

    # list_OutLayer_grad_exp = ["Linear", "Linear_no_bias", "CosLayer", "WeightNorm",
    #                             "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked", "WeightNorm_Masked"]

    list_OutLayer_no_LR = ["KNN", "SLDA", "MeanLayer", "MedianLayer"]
    list_dataset = ["CIFAR10", "Core50", "Core10Lifelong", "Core10Mix","CUB200", "CIFAR100Lifelong"]
    list_pretrained_on = ["CIFAR100", "ImageNet", "ImageNet", "ImageNet","ImageNet", "CIFAR100"]
    list_architecture = [None, "resnet", "resnet", "resnet","resnet", None]
    list_num_tasks = [5, 10, 8, 50, 10, 5]
    list_subset = [None]
    list_seed = [0, 1, 2, 3, 4, 5, 6, 7]
    lr=0.1
    # #
    # list_dataset = ["CIFAR100Lifelong"]
    # list_pretrained_on = ["CIFAR100"]
    # list_architecture = [None]
    # list_num_tasks = [5]

    for dataset, pretrained_on, num_tasks, architecture in zip(list_dataset, list_pretrained_on, list_num_tasks,
                                                               list_architecture):
        full_summary_list, full_config_list, full_name_list = [], [], []

        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks=num_tasks,
                                                                      list_subset=list_subset,
                                                                      list_OutLayer=list_OutLayer_grad_exp,
                                                                      list_seed=list_seed, lr=lr,
                                                                      architecture=architecture)
        full_summary_list = summary_list
        full_config_list =  config_list
        full_name_list = name_list

        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks=num_tasks,
                                                                      list_subset=list_subset,
                                                                      list_OutLayer=list_OutLayer_no_LR,
                                                                      list_seed=list_seed, lr=0.002,
                                                                      architecture=architecture)
        full_summary_list = full_summary_list + summary_list
        full_config_list = full_config_list + config_list
        full_name_list = full_name_list + name_list
        tot_list_OutLayer = list_OutLayer_no_LR + list_OutLayer_grad_exp

        plot_experiment(str_data_type="test accuracy", name_list=full_name_list, config_list=full_config_list, dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=tot_list_OutLayer,
                        list_seed=list_seed, lr=lr,
                        architecture=architecture, name_extension="Grad_Exp")
        plot_experiment(str_data_type="test accuracy", name_list=full_name_list, config_list=full_config_list,
                        dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=tot_list_OutLayer,
                        list_seed=list_seed, lr=lr,
                        architecture=architecture, name_extension="Grad_Exp", legend=True)

####################################################################################
# Forgetting On Lifelong scenarios
####################################################################################
if False:
    list_subset = [None]
    list_seed_cifar = [0, 1, 2, 3, 4, 5, 6, 7]
    list_seed_core = [8, 9, 10, 11, 12, 13, 14]
    lr = 0.1

    list_OutLayer_001 = ["Linear", "Linear_no_bias", "Linear_Masked", "Linear_no_bias_Masked"]
    list_OutLayer_01 = ["CosLayer", "WeightNorm", "OriginalWeightNorm", "CosLayer_Masked", "WeightNorm_Masked", "OriginalWeightNorm_Masked"]

    list_dataset = ["CIFAR100Lifelong", "Core10Lifelong"]
    list_pretrained_on = ["CIFAR100", "ImageNet"]
    list_num_tasks = [5, 8]
    architectures = [None, "resnet"]

    # reset file
    textfile = open("table.txt", "w")
    a = textfile.write("")
    textfile.close()

    for dataset, pretrained_on, num_tasks, architecture in zip(list_dataset, list_pretrained_on, list_num_tasks, architectures):
        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_01,
                                                                      list_seed, lr=0.1, architecture=architecture)

        full_summary_list = summary_list
        full_config_list = config_list
        full_name_list = name_list

        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_001,
                                                                      list_seed, lr=0.01, architecture=architecture)

        full_summary_list += summary_list
        full_config_list += config_list
        full_name_list += name_list
        tot_list_OutLayer = list_OutLayer_001 + list_OutLayer_01
        if dataset == "Core10Lifelong":
            list_seed = list_seed_core
        else:
            list_seed = list_seed_cifar


        plot_experiment(str_data_type="test accuracy task 0.0", name_list=full_name_list, config_list=full_config_list,
                        dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=tot_list_OutLayer,
                        list_seed=list_seed, lr=lr,
                        architecture=architecture, name_extension="Forg_Exp")
        plot_experiment(str_data_type="test accuracy task 0.0", name_list=full_name_list, config_list=full_config_list,
                        dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=tot_list_OutLayer,
                        list_seed=list_seed, lr=lr,
                        architecture=architecture, name_extension="Forg_Exp", legend=True)

####################################################################################
# Subset Experiments
####################################################################################
if False:

    list_subset = [100, 200, 500, 1000, None]
    list_seed = [0, 1, 2, 3, 4, 5, 6, 7]
    lr = 0.1

    list_OutLayer_001 = ["Linear", "Linear_no_bias", "Linear_Masked", "Linear_no_bias_Masked"]
    list_OutLayer_01 = ["CosLayer", "WeightNorm", "OriginalWeightNorm", "CosLayer_Masked", "WeightNorm_Masked", "OriginalWeightNorm_Masked"]
    list_OutLayer_no_LR = ["KNN", "SLDA", "MeanLayer", "MedianLayer"]

    list_dataset = ["CIFAR10", "CIFAR100Lifelong", "Core50", "Core10Lifelong", "CUB200"]
    list_pretrained_on = ["CIFAR100","CIFAR100","ImageNet", "ImageNet", "ImageNet"]
    list_num_tasks = [1, 1, 1, 1, 1]
    architectures = [None, None, "resnet", "resnet", "resnet"]

    # paper parameter
    # list_dataset = ["Core50"]
    # list_pretrained_on = ["ImageNet"]
    # list_num_tasks = [1]
    # architectures = ["resnet"]


    # reset file
    textfile = open("table.txt", "w")
    a = textfile.write("")
    textfile.close()

    for dataset, pretrained_on, num_tasks, architecture in zip(list_dataset, list_pretrained_on, list_num_tasks, architectures):
        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_01,
                                                                      list_seed, lr=0.1, architecture=architecture)

        full_summary_list = summary_list
        full_config_list = config_list
        full_name_list = name_list

        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer_001,
                                                                      list_seed, lr=0.01, architecture=architecture)

        full_summary_list += summary_list
        full_config_list += config_list
        full_name_list += name_list

        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks=num_tasks,
                                                                      list_subset=list_subset,
                                                                      list_OutLayer=list_OutLayer_no_LR,
                                                                      list_seed=list_seed, lr=0.002,
                                                                      architecture=architecture)
        full_summary_list = full_summary_list + summary_list
        full_config_list = full_config_list + config_list
        full_name_list = full_name_list + name_list
        tot_list_OutLayer = list_OutLayer_01 + list_OutLayer_001 + list_OutLayer_no_LR

        # plot_experiment_subset(str_data_type="test accuracy", name_list=name_list, config_list=config_list,
        #                        dataset=dataset,
        #                        pretrained_on=pretrained_on,
        #                        num_tasks=num_tasks, list_subset=list_subset, list_OutLayer=list_OutLayer,
        #                        list_seed=list_seed, architecture=None)
        generate_table_subset(full_name_list, full_summary_list, full_config_list, "test accuracy", dataset, pretrained_on,
                              num_tasks, list_subset, tot_list_OutLayer, list_seed, lr, architecture=architecture)


if False:
    ####################################################################################
    # Continual Experiments
    ####################################################################################

    list_dataset = ["CIFAR100"]
    list_pretrained_on = ["CIFAR10"]
    list_num_tasks = [10]

    list_dataset = ["CIFAR10", "CIFAR100Lifelong", "Core50", "Core10Lifelong", "CUB200"]
    list_pretrained_on = ["CIFAR100", "CIFAR100Lifelong", "ImageNet", "ImageNet", "ImageNet"]
    list_num_tasks = [5, 5, 10, 8, 10]

    # list_dataset = ["CUB200"]
    # list_pretrained_on = ["ImageNet"]
    # list_num_tasks = [10]

    for dataset, pretrained_on, num_tasks in zip(list_dataset, list_pretrained_on, list_num_tasks):
        summary_list, config_list, name_list = select_all_experiments(dataset, pretrained_on, num_tasks, list_subset,
                                                                      list_OutLayer,
                                                                      list_seed, architecture=None)

        nb_exps = len(summary_list)

        plot_experiment(str_data_type="test accuracy", name_list=name_list, config_list=config_list, dataset=dataset,
                        pretrained_on=pretrained_on,
                        num_tasks=num_tasks, subset=list_subset[0], list_OutLayer=list_OutLayer, list_seed=list_seed,
                        architecture=None)



