

# Regulatization Shortcomings for Continual Learning

To reproduce the experiments of the paper:
you need pytorch and torchvision.

version:
commit ee535aef6a98562fe1aca4b15bd70eec28f7094e 

you need to install an anaconda environment:

``
conda env create -f ECML_PKDD.yml
``

Then, you can run the experiments in a terminal with: (it might be several hours long maybe 24h)

``
chmod +x experiment.sh
./experiment.sh
``

Once the experiments are done you can plot figures with: (tested with python 3.8)

``
python Plot_Figures_ECML_PKDD.py
``

The figures will be in "SingleH" and "MultiH" folders.