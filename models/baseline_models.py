import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_EI, max_EI #using EI as acq. function for now; can adjust

import warnings

def get_h_best(initial_X, initial_y, X, y, n_iterations = 50):
    """
    baseline BO code
    """
    # defining the kernel for the Gaussian process
    kernel = Matern(length_scale=1.0)
    regressor = GaussianProcessRegressor(kernel=kernel)

    X_pool, y_pool = X.copy(), y.copy()

    # initializing the optimizer
    optimizer = BayesianOptimizer(
        estimator=regressor,
        X_training=initial_X, y_training=initial_y,
        query_strategy=max_EI
    )

    # Bayesian optimization
    h_pred = []
    h_best = [np.max(initial_y)]

    warnings.filterwarnings('ignore') #for higher n_query > 250, it returns ConvergenceWarning: lbfgs failed to converge (status=2)

    for n_query in range(n_iterations):
        if n_query % 100 == 0:
            print(f'Iteration num. {n_query + 1}')

        query_idx, query_inst = optimizer.query(X_pool)
        optimizer.teach(X_pool[query_idx], y_pool[query_idx])
        h_pred.append(y_pool[query_idx].item())

        if y_pool[query_idx].item() > h_best[-1]:
            h_best.append(y_pool[query_idx].item())
        else:
            h_best.append(h_best[-1])

        # Remove the queried instance from the pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
    
    return h_best, h_pred

def get_rank_best(initial_X, initial_y, X, y, FOM_to_rank, n_iterations = 50):
    """
    get best rank obtained from heat transfer coeff.
    """
    # defining the kernel for the Gaussian process
    kernel = Matern(length_scale=1.0)
    regressor = GaussianProcessRegressor(kernel=kernel)

    X_pool, y_pool = X.copy(), y.copy()
    # initializing the optimizer
    optimizer = BayesianOptimizer(
        estimator=regressor,
        X_training=initial_X, y_training=initial_y,
        query_strategy=max_EI
    )

    # Bayesian optimization
    h_pred = []
    h_best = [np.max(initial_y)]
    rank_best = [FOM_to_rank[np.max(initial_y)]]

    warnings.filterwarnings('ignore') #for higher n_query > 250, it returns ConvergenceWarning: lbfgs failed to converge (status=2)

    for n_query in range(n_iterations):
        if n_query % 100 == 0:
            print(f'Iteration num. {n_query + 1}')

        query_idx, query_inst = optimizer.query(X_pool)
        optimizer.teach(X_pool[query_idx], y_pool[query_idx])
        h_pred.append(y_pool[query_idx].item())

        if y_pool[query_idx].item() > h_best[-1]:
            h_best.append(y_pool[query_idx].item())
            rank_best.append(FOM_to_rank[y_pool[query_idx].item()])
        else:
            h_best.append(h_best[-1])
            rank_best.append(rank_best[-1])

        # Remove the queried instance from the pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
    
    return rank_best

def random_sampling(X, y, n_iterations = 50):
    #taken from baseline_random.ipynb for baseline random sampling model
    best_h_rand = []
    #n_iterations = 500
    randIndices = np.random.choice(len(X), size = len(X), replace = False)

    for n_query in range(n_iterations):
        if n_query % 100 == 0:
            print(f'Iteration num. {n_query + 1}')
        
        indices = randIndices[n_query]
        if n_query == 0:
            best_h_rand.append(y[indices])
        else:
            if y[indices] > best_h_rand[-1]:
                best_h_rand.append(y[indices])
            else:
                best_h_rand.append(best_h_rand[-1])
    
    return best_h_rand