

def check_one_config_parameter(config_parameter, value_or_list):
    if isinstance(value_or_list, list):
        parameter_ok = (config_parameter in value_or_list)
    else:
        parameter_ok = (config_parameter == value_or_list)
    return parameter_ok

def select_run(dict_config,
               scenario_name,
               name_algo,
               dataset,
               pretrained_on,
               num_tasks,
               OutLayer,
               subset,
               seed,
               lr=0.002,
               architecture=None,
               finetuning=False,
               test_label=False,
               spurious_corr=0.0,
               nb_samples_rehearsal_per_class = 100):

    if not check_one_config_parameter(dict_config["scenario_name"], scenario_name): return False
    if not check_one_config_parameter(dict_config["name_algo"], name_algo): return False
    if not check_one_config_parameter(dict_config["dataset"], dataset):  return False
    if not check_one_config_parameter(dict_config["pretrained_on"], pretrained_on):  return False
    if not check_one_config_parameter(dict_config["num_tasks"], num_tasks):  return False
    if not check_one_config_parameter(dict_config["OutLayer"], OutLayer):  return False
    if not check_one_config_parameter(dict_config["subset"], subset):  return False
    if not check_one_config_parameter(dict_config["seed"], seed):  return False
    if not check_one_config_parameter(dict_config["lr"], lr):  return False
    if not check_one_config_parameter(dict_config["finetuning"], finetuning):  return False
    if not check_one_config_parameter(dict_config["test_label"], test_label):  return False

    if "spurious_corr" in dict_config.keys():
        if not check_one_config_parameter(dict_config["spurious_corr"], spurious_corr):  return False
        if not check_one_config_parameter(dict_config["nb_samples_rehearsal_per_class"], nb_samples_rehearsal_per_class):  return False

    if dataset in ["Core50", "Core10Lifelong", "Core10Mix", "CUB200"]:
        architecture_ok = check_one_config_parameter(dict_config["architecture"], architecture)
    else:
        architecture_ok = True

    return architecture_ok