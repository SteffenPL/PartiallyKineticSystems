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



n = 100

# masses
m_r = 2.
m_q = 10. / n

# stiffness
kappa_r = 1.
kappa_q = 1 / n

# forces
U_r = lambda r, dr: 0.5 * kappa_r * r**2
T_r = lambda r, dr: 0.5 * m_r * dr**2

U_q = lambda q, dq: 0.5 * kappa_q * q**2
T_q = lambda q, dq: 0.5 * m_q * dq**2


# constraint
g = lambda r, q : q - r



# initial distribution
r0 = 1.
dr0 = 0.
Q0 = np.random.normal(loc=2., scale=1., size=(n,))
t_end = 60


dpms.init_equations(T_r, U_r, T_q, U_q, g)
dpms.init_state(r0, dr0, Q0, t_end, n_eval=1000)

G_noise = np.concatenate( [np.array([0.]), np.ones((n+1,))] )

dpms.simulate(G=lambda y,t: np.diag(G_noise))


path = "../../../documents/paper/images/"
fname = path + "linear_stoch"



dpms.plot_g(levels=100)
plt.savefig(fname + "_contour.pdf")
plt.show()

dpms.plot_particle_paths()
plt.savefig(fname + "_particles_time.pdf")
plt.show()

dpms.plot_g_img(alpha=0.4)
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
plt.savefig(fname + "forces.pdf")
plt.show()












