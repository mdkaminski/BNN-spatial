# Bayesian Neural Network Priors for Making Inference from Spatial Data

Code for my Honours thesis, titled _"Bayesian Neural Network Priors for Making Inference from Spatial Data"_.

## Terminology

- GP: Gaussian Process
- BNN: Bayesian Neural Network
- GPi-G: GP-induced BNN prior with Gaussian weights and biases
- SST: Sea-Surface Temperature

## Examples

The folder `scripts` contains several `Python` script files, implementing the examples used in the Honours thesis.

- `Ch4_BNN_GPiG.py` implements the 1D BNN regression example used in Chapter 4.
- `Ch5_1_BNN_GPiG_2D.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.1), when calibrating a BNN prior _with an embedding layer_ to a stationary target GP prior.
- `Ch5_1_BNN_GPiG_2D_Naive.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.1), when calibrating a rudimentary BNN prior _without an embedding layer_ to a stationary target GP prior.
- `Ch5_3_BNN_GPiG_NS.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.3), when calibrating a BNN prior _with spatially-dependent hyperparameters_ to a nonstationary target GP prior. Posterior inference is performed with Bayesian model averaging.
- `Ch5_3_BNN_GPiG_NS_Naive.py` implements a 2D BNN regression example used in Chapter 5 (Section 5.3), when calibrating a rudimentary BNN prior _with spatially-invariant hyperparameters_ to a nonstationary target GP prior.
- `Ch6_SST_GPiG.py` implements a BNN calibration and regression example used in Chapter 6, when calibrating a BNN prior _with spatially-dependent hyperparameters_ to an empirical SST process. Posterior inference is performed with Bayesian model averaging.
- `Ch6_SST_GPiG_Stat.py` implements a BNN calibration and regression example used in Chapter 6, when calibrating a rudimentary BNN prior _with spatially-invariant hyperparameters_ to an empirical SST process.
