

import wandb


api = wandb.Api()
NAME_PROJECT = "tlesort/CLOOD"
runs = api.runs(NAME_PROJECT)

LIST_ARGUMENTS = ["seed", "scenario_name", "num_tasks", "dataset","name_algo", "lr", "OutLayer", "nb_samples_rehearsal_per_class", "spurious_corr"]
# TO ADD STILL :  root_dir, data_dir


list_commands = []

counter = 0
for run in runs:
    if run.state != "finished":

        if run.config["seed"]==1664:
            continue
        command = ''
        for argument in LIST_ARGUMENTS:
            command += f" --{argument} {run.config[argument]}"


        list_commands.append(command)

len_before = len(list_commands)
print("remove duplicate entries")
list_commands = list( dict.fromkeys(list_commands) )
print(f"{len_before-len(list_commands)} duplicate entries removed")

commands = '#!/bin/bash\n\n'
for command in list_commands:
    commands += f"python main.py {command}  --fast $@ \n\n"

f = open("failed_runs.sh", "w")
f.write(commands)
f.close()