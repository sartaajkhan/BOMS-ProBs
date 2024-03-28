"""
Constructed to generate initial_X and initial_y sets for BO group
Methodology used: random sampling w/o replacement
-Can consider Latin-Hypercube sampling
"""

from scipy.stats import qmc
import numpy as np
import pandas as pd

df = pd.read_excel("Dataset.xlsx")
df_interest = df[['Viscosity (cP)', 'Density (kg/m^3)', 'Thermal Conductivity (W/m.k)', 'Heat Capacity (J/(kg.K))', 'Heat Transfer Coefficient']]

X = df_interest[['Viscosity (cP)', 'Density (kg/m^3)', 'Thermal Conductivity (W/m.k)', 'Heat Capacity (J/(kg.K))']].to_numpy() #full X
y = df_interest['Heat Transfer Coefficient'].to_numpy() #full y

def create_initial_set(X_, y_, num_samples = 10):
    assert len(X_) == len(y_)
    indices = np.random.choice(len(X_), num_samples, replace=False) #randomly sampling X indices w/o replacement
    print(indices)
    initial_X, initial_y = X_[indices], y_[indices]

    return initial_X, initial_y

def return_indices(X_, y_, num_samples = 10):
    """
    more consistent and transferrable
    """
    assert len(X_) == len(y_)
    indices = np.random.choice(len(X_), num_samples, replace = False) #randomly sampling X indices w/o replacement
    return indices

setting = 'indices' #can change between 'indices' and 'direct' (direct returns initial_X and initial_y)

if __name__ == "__main__":
    if setting == 'direct':
        initial_X, initial_y = create_initial_set(X, y, num_samples=10)
        with open('initial_X.npy', 'wb') as f:
            np.save(f, initial_X)

        with open('initial_y.npy', 'wb') as f:
            np.save(f, initial_y)
    
    elif setting == 'indices':
        indices = return_indices(X, y, num_samples=10)
        with open('initial_indices.npy', 'wb') as f:
            np.save(f, indices)
        
    print("Sets generated. Please check folder.")