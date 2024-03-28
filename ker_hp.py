# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI, max_EI #using EI as acq. function for now; can adjust

import warnings

kernel_array_stand_alone = []
kernel_array_stand_alone_par = []

from sklearn.gaussian_process.kernels import DotProduct # k = sigma_0 + x_i*x_j
for sigma_0_i in np.geomspace(0.01, 100, num=8):
    kernel_array_stand_alone.append(DotProduct(sigma_0=sigma_0_i))
    kernel_array_stand_alone_par.append(["DotProduct", sigma_0_i])

# from sklearn.gaussian_process.kernels import ExpSineSquared # k = exp( (-2sin^2(pi*d(x_i,x_j)/p)) / (l^2) )
# for l_i in np.geomspace(0.01, 100, num=8):
#     for p_i in np.geomspace(0.01, 100, num=8):
#         kernel_array_stand_alone.append(ExpSineSquared(length_scale= l_i, periodicity= p_i))
#         kernel_array_stand_alone_par.append(["ExpSineSquared", [l_i, p_i]])

from sklearn.gaussian_process.kernels import Matern
for l_j in np.geomspace(0.01, 100, num=8):
    for nu_i in [0.5, 1.5, 2.5]:
        kernel_array_stand_alone.append(Matern(length_scale= l_j, nu= nu_i))
        kernel_array_stand_alone_par.append(["Matern", [l_j, nu_i]])

from sklearn.gaussian_process.kernels import RBF # k = exp( (-d(x_i, x_j)^2) / (2*l^2))
for l_k in np.geomspace(0.01, 100, num=8):
    kernel_array_stand_alone.append(RBF(length_scale= l_k))
    kernel_array_stand_alone_par.append(["RBF", l_k])

from sklearn.gaussian_process.kernels import RationalQuadratic # k = (1+(d(x_i, x_j)^2)/(2*alpha*l^2))^(-alpha)
for l_ii in np.geomspace(0.01, 100, num=8):
    for alpha_i in np.geomspace(0.01, 100, num=8):
       kernel_array_stand_alone.append(RationalQuadratic(length_scale= l_ii, alpha= alpha_i))
       kernel_array_stand_alone_par.append(["RationalQuadratic", [l_ii, alpha_i]]) 

from sklearn.gaussian_process.kernels import WhiteKernel # k = noise_level if x_i = x_j, 0 ow
for noise_level_i in np.geomspace(0.01, 100, num=8):
    kernel_array_stand_alone.append(WhiteKernel(noise_level= noise_level_i))
    kernel_array_stand_alone_par.append(["WhiteKernel", noise_level_i])
