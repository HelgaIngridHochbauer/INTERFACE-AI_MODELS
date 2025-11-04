# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:43:55 2025

@author: USER
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


folder = "C:\\Users\\USER\\Documents\\Research\\AI4MultiGIS\\Old\\all_AS\\arqueoastro\\anglenet" #update accordingly

vals = np.loadtxt(os.path.join(folder, 'classification_output-predicted.txt')) # plot the predicted angles

vals_pred = vals*1 

plt.hist(vals_pred, bins=360)

plt.show()


folder = "C:\\Users\\USER\\Documents\\Research\\AI4MultiGIS\\Old\\all_AS\\arqueoastro\\database\\orientations" #update accordingly

vals_real = pd.read_csv('C:\\Users\\USER\\Documents\\Research\\AI4MultiGIS\\Old\\all_AS\\arqueoastro\\database\\orientations\\angles.txt', delimiter=' ', header=None, usecols=[3]) # plot the actual angles

plt.hist(vals_real, bins=360)

plt.show()

print (list(vals_real))

diff = abs(list(vals_real) - vals_pred)


plt.hist(diff, bins=360) # plot the difference


plt.show()


