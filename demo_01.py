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

init_plot_settings(plt)


dpms = DiscretePMS()



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
g = lambda r, q : -factor * sp.sin(0.1*q)/(1+r**2) + 0.1*q
# g = lambda r, q : q**2 - 1/(1+(r-0.5)**2)   # start within the circle!
# g = lambda r, q : q - r
# g = lambda r, q : q**2 + r**2  # looks nice and smooth!
# g = lambda r, q : 



# initial distribution
r0 = 1.
dr0 = 0.
Q0 = np.random.normal(loc=0.9, scale=1., size=(n,))
t_end = 60


dpms.init_equations(T_r, U_r, T_q, U_q, g)
dpms.init_state(r0, dr0, Q0, t_end)
dpms.simulate()


path = "../../../documents/paper/images/"
fname = path + "demo"



dpms.plot_g(levels=100)
plt.savefig(fname + "_contour.pdf")
plt.show()

dpms.plot_particle_paths()
plt.savefig(fname + "_particles_time.pdf")
plt.show()

dpms.plot_particle_paths(use_r_axis=True,plot_singular_pts=True)
plt.savefig(fname + "_particles_statespace.pdf")
plt.show()

dpms.plot_heavy_system()
plt.savefig(fname + "_heavy.pdf")
plt.show()

dpms.calc_energies(show_plot=True)
plt.savefig(fname + "_energies.pdf")
plt.show()


dpms.calc_mod_mass_force(show_plot=True)
plt.show()













