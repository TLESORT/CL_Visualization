
Configs files to run HPS sweep

```
wandb sweep --name baseline_sweep -p CLOOD Configs/config_files.yaml 
```



preliminaries_ib_irm ~ 100 runs

```
wandb sweep --name baseline_ib_irm_sweep -p CLOOD Configs/preliminaries_ib_irm.yaml 
```


preliminaries_ib_erm ~ 100 runs

```
wandb sweep --name baseline_ib_erm_sweep -p CLOOD Configs/preliminaries_ib_erm.yaml 
```