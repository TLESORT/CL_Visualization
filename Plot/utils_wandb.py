

def check_one_config_parameter(config_parameter, value_or_list):
    if isinstance(value_or_list, list):
        parameter_ok = (config_parameter in value_or_list)
    else:
        parameter_ok = (config_parameter == value_or_list)
    return parameter_ok

def select_run(dict_config, dataset, pretrained_on, num_tasks, OutLayer, subset, seed, lr=0.002, architecture=None):
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


    lr_ok = check_one_config_parameter(dict_config["lr"], lr)
    if not lr_ok: return False

    if dataset in ["Core50", "Core10Lifelong"]:
        architecture_ok = check_one_config_parameter(dict_config["architecture"], architecture)
    else:
        architecture_ok = True
    return architecture_ok