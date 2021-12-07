

import wandb


api = wandb.Api()
NAME_PROJECT = "tlesort/CLOOD"
runs = api.runs(NAME_PROJECT)

LIST_ARGUMENTS = ["seed", "scenario_name", "num_tasks", "dataset","name_algo", "lr", "OutLayer", "nb_samples_rehearsal_per_class", "spurious_corr"]
# TO ADD STILL :  root_dir, data_dir

commands = ''

counter = 0
for run in runs:
    if run.state != "finished":

        if run.config["seed"]==1664:
            continue

        commands += "python main.py"
        for argument in LIST_ARGUMENTS:
            commands += f" --{argument} {run.config[argument]}"

        commands += " --fast $@ \n\n"

f = open("failed_runs.sh", "w")
f.write(commands)
f.close()