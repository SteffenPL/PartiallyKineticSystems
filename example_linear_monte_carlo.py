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


dpms = DiscretePMS()
dpms.name = "linear_mc"

n_particles = 2**np.arange(2,11)
mean_vec = []
var_vec = []

n_mc = 100

t_end = 36
r0 = 1.
dr0 = 0.

mean_field_plot = True
var_analysis = True

mean_trajs = {}
all_trajs = {}


for n in n_particles:

    
    # masses
    m_r = 20.
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
    g = lambda r, q : q + r
    
    
    r_end = np.zeros( shape=(n_mc,) )
    
    
    dpms.init_equations(T_r, U_r, T_q, U_q, g)
    
    
    # colors in plots
    col_min = np.array([0.4, 0.7, 1.])
    col_max = np.array([0.7, 0.4, 1.])
    
    
    mean_r = np.zeros((1000,))
    for i in range(0,n_mc):
        
        # initial distribution
        Q0 = np.random.normal(loc=2., scale=1., size=(n,))
        
        dpms.init_state(r0, dr0, Q0, t_end, n_eval=1000)
        dpms.simulate()
    
        r_end[i] = dpms.r[-1]
        p = i/(n_mc-1)
        
        plt.plot(dpms.sol.t, dpms.r, color=p * col_min + (1.-p) * col_max, lw=0.3)
        
        mean_r += 1./n_mc * dpms.r
    
    # calucate mean and variance at t_end
    
    mean_trajs[n] = mean_r
    
    cur_mean = 1/n_mc * np.sum( r_end )
    cur_var = 1/(n_mc-1) * np.sum( (r_end - cur_mean)**2  ) 

    
    plt.plot(dpms.sol.t, mean_r, color='black', lw=2)
        
    mean_vec.append(cur_mean)
    var_vec.append(cur_var)
    
    
    plt.xlabel(r"$t$")
    plt.ylabel(r"$r(t)$")
    plt.title("Macroscopic System (DAE model) \n (%d samples, N = %d cross-bridges)" % (n_mc,n))
    plt.savefig("simulations/%s%d_%d.eps" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.savefig("simulations/%s%d_%d.png" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.show()

bk_mean_trajs = mean_trajs

  



if var_analysis:
    
    plt.loglog(n_particles, var_vec, label=r"$\mathrm{Var}[ r(t_\mathrm{end}) ]$")
    plt.loglog(n_particles, 1./n_particles, ":", label=r"$N^{-1}$")
    plt.xlabel(r"$\log(N)$")
    #plt.ylabel(r"$\log$")
    plt.legend()
    plt.savefig("simulations/var_%s%d.eps" % (dpms.name, n_mc), bbox_inches = 'tight')
    plt.savefig("simulations/var_%s%d.png" % (dpms.name, n_mc), bbox_inches = 'tight')
    plt.savefig("simulations/var_%s%d.svg" % (dpms.name, n_mc), bbox_inches = 'tight')
    plt.show()
    
    #save_plots( plt, dpms, "disc")


if mean_field_plot:
    
    mean_trajs_sub = {}
    mean_trajs_sub[8] = mean_trajs[8]
    mean_trajs_sub[64] = mean_trajs[64]
    mean_trajs_sub[256] = mean_trajs[256]
    mean_trajs_sub[1024] = mean_trajs[1024]
    
    pms = DiscretePMS()
    
    n = 250
    # masses
    m_r = 20.
    m_q = 10. / n
    
    # stiffness
    kappa_r = 1.
    kappa_q = 1 / n
    
    # forces
    U_r = lambda r, dr: 0.5 * kappa_r * r**2
    T_r = lambda r, dr: 0.5 * m_r * dr**2
    
    U_q = lambda q, dq: 0.5 * kappa_q * q**2
    T_q = lambda q, dq: 0.5 * m_q * dq**2

    # generate grid
    loc = 2.
    scale = 1
    qgrid_size = 100
    qgrid = np.linspace(-5,7, qgrid_size)
    rho0 = n * norm.pdf( qgrid, loc=loc, scale=scale )
    
    
    pms.init_equations(T_r, U_r, T_q, U_q, g)
    pms.init_meso(r0, dr0, rho0, qgrid, t_end, n_eval=1000)
    pms.simulate_meso(method="Radau",atol=1.e-8,rtol=1.e-8,use_upwind=True)
    

    plt.plot(pms.sol.t, pms.r, color=[1.0,0.6,.1],lw=3)
    
    # colors in plots
    col_min = np.array([.0, 0.0, 0.])
    col_max = np.array([0.2, 0.5, 0.2])
    
    n_sims = len(mean_trajs_sub.keys())
    n_dots = 50
    f_start = 0
    markers = ["o","o",".","x","+"]
    i = 0
    for (n, mean_r) in mean_trajs_sub.items():
        p = i/(n_sims-1.)
        plt.plot(dpms.sol.t[f_start*i:1000:n_dots], mean_r[f_start*i:1000:n_dots], markersize=10,color = p * col_min + (1.-p) * col_max, lw=0.0, marker=markers[i])
        i += 1
    
    
    
    plt.xlabel(r"$t$")
    plt.ylabel(r"$r(t)$")
    plt.title("Macropscopic System (Mean-Field Limit)")
    plt.legend(["kinetic"] + [ "mean for N = %d"%n for n in  mean_trajs_sub.keys()],
                    loc='lower left', bbox_to_anchor=(0.,-0.3),
                    bbox_transform=plt.gcf().transFigure, ncol=2)
    
    
    i = 0
    for (n, mean_r) in mean_trajs_sub.items():
        p = i/(n_sims-1.)
        plt.plot(dpms.sol.t, mean_r, color = p * col_min + (1.-p) * col_max, lw=0.5)
        i += 1
    
    
    plt.savefig("simulations/meanfield_%s%d_%d.png" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.savefig("simulations/meanfield_%s%d_%d.eps" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.savefig("simulations/meanfield_%s%d_%d.svg" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.show()

    mean_trajs_sub = {}
    mean_trajs_sub[4] = mean_trajs[4]
    mean_trajs_sub[8] = mean_trajs[8]
    mean_trajs_sub[16] = mean_trajs[16]
    mean_trajs_sub[32] = mean_trajs[32]
    mean_trajs_sub[64] = mean_trajs[64]
    mean_trajs_sub[128] = mean_trajs[128]
    mean_trajs_sub[256] = mean_trajs[256]
    mean_trajs_sub[512] = mean_trajs[512]
    mean_trajs_sub[1024] = mean_trajs[1024]

    plt.loglog( [n for n in mean_trajs_sub.keys()], 
                [np.max( np.abs( mean_r - pms.r) ) for mean_r in mean_trajs_sub.values()]  )

    plt.xlabel(r"$N$ (Number of Cross-Bridges)")
    plt.ylabel("$\max_t | \mathbf{E}[r^{\mathrm{DAE}}(t)] - r^{\mathrm{kin}}(t) |$")
    plt.title("Macroscopic System (Mean-Field Limit)")
    plt.savefig("simulations/meanfield_error_%s%d_%d.eps" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.savefig("simulations/meanfield_error_%s%d_%d.png" % (dpms.name, n_mc, n), bbox_inches = 'tight')
    plt.savefig("simulations/meanfield_error_%s%d_%d.svg" % (dpms.name, n_mc, n), bbox_inches = 'tight')





