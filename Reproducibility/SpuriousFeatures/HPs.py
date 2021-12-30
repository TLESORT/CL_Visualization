


def get_selected_HPs_Spurious(config):
    """"This function gives HPs selected through baseline sweeps for final experiments."""
    if config.dataset == "CIFAR10" and config.scenario_name == "SpuriousFeatures":
        if config.name_algo == "irm":
            # https://wandb.ai/tlesort/CLOOD/runs/j6plvbvy?workspace=user-tlesort
            config.opt = "Adam"
            config.irm_lambda = 10
            config.irm_penalty_anneal_iters = 1
            config.lr = 0.02198
        elif config.name_algo == "ib-irm":
            # https://wandb.ai/tlesort/CLOOD/runs/318ctjmb?workspace=user-tlesort
            config.opt = "Adam"
            config.irm_lambda = 1000
            config.irm_penalty_anneal_iters = 1000
            config.ib_lambda = 10
            config.ib_penalty_anneal_iters = 10000
            config.lr = 0.08116
        elif config.name_algo == "ib-erm":
            # https://wandb.ai/tlesort/CLOOD/runs/8bt8b5v6?workspace=user-tlesort
            config.opt = "Adam"
            config.ib_lambda = 10
            config.ib_penalty_anneal_iters = 1
            config.lr = 0.02565
        elif config.name_algo == "rehearsal" or config.name_algo == "baseline":
            # https://wandb.ai/tlesort/CLOOD/runs/k0f6y6h8?workspace=user-tlesort
            config.opt = "Adam"
            config.lr = 0.05906
        else:
            raise ValueError("This config is not ready to be tested.")
    else:
        raise ValueError("This config is not ready to be tested.")

    print("Update Config with Sweeps HPs")
    return config
