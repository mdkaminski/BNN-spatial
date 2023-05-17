# Bayesian Neural Network Priors for Making Inference from Spatial Data

This GitHub page provides code for reproducing the results in my Honours thesis, titled _"Bayesian Neural Network Priors for Making Inference from Spatial Data"_. This thesis develops a framework for calibrating Bayesian neural network priors to generic spatial processes, and for performing subsequent posterior inference (regression) starting from the calibrated prior. This framework is developed with the ultimate aim of implementation on sea-surface temperature data, extracted from the North Atlantic ocean.

## Terminology

- GP: Gaussian Process
- BNN: Bayesian Neural Network
- GPi-G: **GP-i**nduced BNN prior with **G**aussian weights and biases
- SST: Sea-Surface Temperature

## Examples

The folder `scripts` contains several `Python` script files, implementing the examples used in the Honours thesis.

- `Ch4_BNN_GPiG.py` implements the 1D BNN regression example used in Chapter 4.
- `Ch5_1_BNN_GPiG_2D.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.1), when calibrating a BNN prior _with an embedding layer_ to a stationary target GP prior.
- `Ch5_1_BNN_GPiG_2D_Naive.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.1), when calibrating a rudimentary BNN prior _without an embedding layer_ to a stationary target GP prior.
- `Ch5_3_BNN_GPiG_NS.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.3), when calibrating a BNN prior _with spatially-dependent hyperparameters_ (and with an embedding layer) to a nonstationary target GP prior.
- `Ch5_3_BNN_GPiG_NS_Naive.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.3), when calibrating a rudimentary BNN prior _with spatially-invariant hyperparameters_ (and with an embedding layer) to a nonstationary target GP prior.
- `Ch6_SST_GPiG.py` implements a BNN calibration and regression example used in Chapter 6, when calibrating a BNN prior _with spatially-dependent hyperparameters_ (and with an embedding layer) to an empirical SST process.
- `Ch6_SST_GPiG_Stat.py` implements a BNN calibration and regression example used in Chapter 6, when calibrating a rudimentary BNN prior _with spatially-invariant hyperparameters_ (and with an embedding layer) to an empirical SST process.

## Software Requirements

Python (>= 3.7), PyTorch, NumPy, SciPy, Matplotlib, Seaborn, pandas

Exact versions used: Python 3.7.13, PyTorch 1.9.0+cu111, NumPy 1.21.5, SciPy 1.7.3, Matplotlib 3.5.3, Seaborn 0.12.1, pandas 1.3.5

Note: The code may not work with the latest versions of PyTorch (> 1.9.0).

## Instructions

The examples in Chapter 6 require the SST data set, available in NetCDF format here:

https://hpc.niasra.uow.edu.au/azm/global-analysis-forecast-phy-001-024_1551608429013.nc

After cloning this repository, create a new `data` folder within the `scripts` folder. Then, place the NetCDF file into the `data` folder. The script files will then be able to access the SST data set.


