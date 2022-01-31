
Configs files to run HPS sweep

```
wandb sweep --name baseline_sweep -p CLOOD Configs/config_files.yaml 
```


# Support Experiments

# Spurious Experiments

```
wandb sweep --name spurious_sweep -p CLOOD Configs/Spurious_exps.yaml 
```

# Preliminary experiments
preliminaries_irm ~ 100 runs

```
wandb sweep --name baseline_irm_sweep -p CLOOD Configs/preliminaries_irm.yaml 
```

preliminaries_ib_irm ~ 100 runs

```
wandb sweep --name baseline_ib_irm_sweep -p CLOOD Configs/preliminaries_ib_irm.yaml 
```


preliminaries_ib_erm ~ 100 runs

```
wandb sweep --name baseline_ib_erm_sweep -p CLOOD Configs/preliminaries_ib_erm.yaml 
wandb agent tlesort/CLOOD/wyngb3xx --count 1
```