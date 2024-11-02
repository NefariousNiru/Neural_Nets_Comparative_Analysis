# This project aims to model 3 datasets
- Auto MPG
- Seoul Bike Share Demand
- Boston Housing

## The objective is to test the model on various regression techniques.
- 2L_NN
- 3L_NN
- 4L_NN
- Custom_NN
- Random Forest Regression
- Feature Selection

# Steps to run
Install Pycharm

Create a virtual environment using Pycharm UI instructions. 

Install package dependencies using
`pip install -r requirements.txt`
*P.S - Pytorch will not be installed this way*

# The structure of the project follows
- datasets (directory containing datasets and preprocessing)

Neural Nets Architectures
    - TwoLayerNN.py
    - ThreeLayerNN.py
    - FourLayerNN.py
    - CustomNN.py

- util (utility methods - pre_processing, loading, metrics)

All NN folders contain `cross_val`, `train_test` and `in_sample` which is the entry point for the application for different validation techinques,
except random forest regression `auto_mpg.py`, `boston.py`, `seoul_bike.py` and feature selection using `run.py` respectively.

# All dependencies are listed `requirements.txt` - Install package dependencies using
`pip install -r requirements.txt`

# Pytorch installation will fail so install using
`pip install torch --index-url https://download.pytorch.org/whl/cu118`