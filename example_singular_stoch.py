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


from init_plot_settings import init_plot_settings
from save_state import save_plots
init_plot_settings(plt)


dpms = DiscretePMS()



n = 50

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
Q0 = np.random.normal(loc=2., scale=0.2, size=(n,))
t_end = 8


dpms.init_equations(T_r, U_r, T_q, U_q, g)
dpms.init_state(r0, dr0, Q0, t_end, n_eval=8000)

G_noise = 0.2 * np.concatenate( [np.array([0.,0.]), np.ones((n,))] )

dpms.simulate(G=lambda y,t: np.diag(G_noise))


save_plots( plt, dpms, "singular_stoch_new", "disc")










