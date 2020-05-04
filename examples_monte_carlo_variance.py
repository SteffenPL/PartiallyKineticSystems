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
from save_state import save_plots, gen_name

init_plot_settings(plt)



def analyse_variance(example_name, n_particles=None, n_mc = 50):
    
    dpms = DiscretePMS()
    dpms.name = example_name
    
    if n_particles is None:
        n_particles = 2**np.arange(2,7)
        
    mean_vec = []
    var_vec = []
    
    for n in n_particles:
    
        if dpms.name == "linear":
            
            
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
            g = lambda r, q : q + r
            
            t_end = 23
            r0 = 1.
            dr0 = 0.
            
            mean = 2.0
            scale = 1.0
        
        if dpms.name == "onion":
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
        
        
            
            t_end = 20
            r0 = 1.
            dr0 = 0.
                
            mean = 0.9
            scale = 1.0
                
            
        if dpms.name == "singular":
                        
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
            
            g = lambda r, q : q**2 + (r-2)**2
            
            # initial distribution
            r0 = 3.
            dr0 = 0.
            #Q0 = np.random.normal(loc=2, scale=0.75, size=(n,))
            t_end = 3.5
            
            mean = 2
            scale = 0.2
            
        r_end = np.zeros( shape=(n_mc,) )
        
        dpms.init_equations(T_r, U_r, T_q, U_q, g)
        
        
        # colors in plots
        col_min = np.array([0., 0.6, 0.9])
        col_max = np.array([0.6, 0., 0.9])
        
        
        for i in range(0,n_mc):
            
            # initial distribution
            
            Q0 = np.random.normal(loc=mean, scale=scale, size=(n,))
            
            dpms.init_state(r0, dr0, Q0, t_end)
            dpms.simulate()
        
            r_end[i] = dpms.r[-1]
            p = i/(n_mc-1)
            
            plt.plot(dpms.sol.t, dpms.r, color=p * col_min + (1.-p) * col_max, lw=0.3)
        
        
        # calucate mean and variance at t_end
        
        
        cur_mean = 1/n_mc * np.sum( r_end )
        cur_var = 1/(n_mc-1) * np.sum( (r_end - cur_mean)**2  ) 
        
        
        mean_vec.append(cur_mean)
        var_vec.append(cur_var)
        
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"$r(t)$")
        plt.title("heavy system (%d samples, %d particles)" % (n_mc,n))
        plt.savefig(gen_name(dpms, "disc") + ("_mc%d_n%d" %( n_mc, n )) + ".pdf")
        plt.show()
    
    
    
    plt.loglog(n_particles, var_vec, label=r"$\mathrm{Var}[ r(t_{\mathrm{end}} ) ]$")
    plt.loglog(n_particles, 1./n_particles, ":", lw=1, label=r"$n^{-1}$")
    plt.xlabel(r"$\log(n)$")
    plt.title("variance of the heavy system")
    plt.ylabel("variance")
    plt.legend()
    plt.savefig(gen_name(dpms, "disc") + ("_variance_mc%d" %( n_mc )) + ".pdf")
    plt.show()
    
    







