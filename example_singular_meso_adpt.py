#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:17:11 2019

@author: plunder
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:44:55 2019

@author: plunder
"""



import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from discrete_pms import DiscretePMS

from scipy.stats import norm

from init_plot_settings import init_plot_settings
from save_state import save_plots

init_plot_settings(plt)


pms = DiscretePMS()



n = 100

# masses
m_r = 2.
m_q = 1. / n

# stiffness
kappa_r = 1.
kappa_q = 0.0 / n

# forces
U_r = lambda r, dr: 0.5 * kappa_r * r**2
T_r = lambda r, dr: 0.5 * m_r * dr**2

U_q = lambda q, dq: 0.5 * kappa_q * q**2
T_q = lambda q, dq: 0.5 * m_q * dq**2


# constraint
factor = 1.
# g = lambda r, q : factor*q/(1+r**2) - q**3
# g = lambda r, q : q**2 - 1/(1+(r-0.5)**2)   # start within the circle!
# g = lambda r, q : q - r
g = lambda r, q : q**2 + (r-2)**2  # looks nice and smooth!
# g = lambda r, q : 



# initial distribution
r0 = 3.
dr0 = 0.
#Q0 = np.random.normal(loc=2, scale=0.75, size=(n,))
t_end = 5

# generate grid
loc = 2
scale = 0.2
qgrid_size = 200
grid_scale = 2
qgrid = np.linspace(-(1**(1/grid_scale)), 4**(1/grid_scale), qgrid_size)
qgrid = qgrid**grid_scale * np.sign(qgrid)**(grid_scale+1)

# remove singularity
qgrid = np.delete(qgrid, np.argwhere(qgrid==0.))
rho0 = n * norm.pdf( qgrid, loc=loc, scale=scale )



pms.init_equations(T_r, U_r, T_q, U_q, g)
pms.init_meso(r0, dr0, rho0, qgrid, t_end, n_eval=1000)
pms.simulate_meso(method='BDF',atol=1.e-6,rtol=1.e-6)


pms.name = "singular_adapt"
save_plots( plt, pms, pms.name, "meso")

#
#
#
#
#
#
#
#
#
#
#
