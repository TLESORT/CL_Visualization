

# Regularization Shortcomings for Continual Learning

To reproduce the experiments of the paper:
you need pytorch and torchvision.

you need to install an anaconda environment:

``
conda env create -f Regularization_Shortcomings.yml
``

Then, you can run the experiments in a terminal with: (it might be several hours long maybe 24h)

``
chmod +x experiment.sh
./experiment.sh
``

Once the experiments are done you can plot figures with: (tested with python 3.8)

``
python Plot_Figures_Regu_Short.py
``

The figures will be in "SingleH" and "MultiH" folders.