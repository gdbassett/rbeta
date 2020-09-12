# Rbeta

This package implements the rbeta algorithm for temporally offset correlation of fMRI BOLD signal signals.  it is based on the matlab code from  Based on the matlab code at https://github.com/remolek/NFC (by Jeremi Ochab) that implements the algorithm described in https://arxiv.org/abs/2007.15728 (Ignacio Cifre, Maria T. Miller Flores, Jeremi K. Ochab, Dante R. Chialvo, "Revisiting non-linear functional brain co-activations: directed, dynamic and delayed", 2019)  Matlab translation by Mayisha Zeb Nakib.  Parallelization by Gabriel Bassett.

It provides the rbeta() function to return the mean correlation as well as individual correlatiosn between a pair of BOLD time series.

it also provides a parallelized function, prbeta(), that only returns the mean rbeta correlations.