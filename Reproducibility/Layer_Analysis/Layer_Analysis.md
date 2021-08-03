

# Continual learning and deep networks: Analysis of the Last Layer

To reproduce the experiments of the paper:
you need essentially python 3.8, pytorch and torchvision and a wandb account.

Move to proper branch:
``
git checkout Layer_Analysis
``

you need to install an anaconda environment:

``
conda env create -f env_layer_analysis.yml
source activate py38
``

Then, you can run the experiments in a terminal with: (it might be several hours/days/week long)

Preliminary experiments
``
chmod +x preliminary_exp.sh
./preliminary_exp.sh
``
Continual experiments
``
chmod +x continual_exp.sh
./continual_exp.sh
``
Preliminary experiments
``
chmod +x subset_exp.sh
./subset_exp.sh
``

Once the experiments are done you can reproduce similar figures to the one in the paper in the wandb environment.
