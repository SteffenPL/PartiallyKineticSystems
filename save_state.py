#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:54:15 2019

@author: plunder
"""

basepath = "./simulations/"  #"../../../documents/paper_august/images/simulations/"


def gen_name(dpms, sim_type):
    name = dpms.name
        
    is_discrete = (sim_type is None or sim_type == "disc")
    
    fname = basepath + name 
    if not is_discrete:
        fname = fname + "_meso"
    
    return fname

def save_plots(plt, dpms, sim_type="", levels=100, nParticles=1000, save_files=True, imgtype = ".png"):
    
    is_discrete = (sim_type is None or sim_type == "disc")
    
    fname = gen_name(dpms, sim_type)
    
    dpms.plot_g(levels=levels)
    plt.savefig(fname + "_contour" + imgtype)
    plt.show()
    
    dpms.plot_veff_img(detail=500)
    plt.savefig(fname + "_veff" + imgtype)
    plt.show()

    
    if is_discrete:
        dpms.plot_particle_paths()
        plt.savefig(fname + "_particles_time" + imgtype)
        plt.show()
        
        #dpms.plot_g(levels=100, alpha=0.3,plot_singular_pts=True, lw=0.3)
        dpms.plot_g_img(alpha=0.4)
        dpms.plot_particle_paths(use_r_axis=True, plot_singular_pts=True, nParticles=nParticles, lw=0.4)
        plt.savefig(fname + "_particles_statespace" + imgtype)
        plt.show()
    else:
        dpms.plot_particle_paths_meso_time()
        plt.savefig(fname + "_particles_time" + imgtype)
        plt.show()

    
    dpms.plot_heavy_system()
    if is_discrete:
        plt.title("Macroscopic System" + " (DAE Model)")
    else:
        plt.title("Macroscopic System" + " (Kinetic Model)")
    plt.savefig(fname + "_heavy" + imgtype)
    plt.show()
    
    if is_discrete:
        dpms.calc_energies(show_plot=True)
    else:
        dpms.calc_energies_meso(show_plot=True)
    
    plt.savefig(fname + "_energies" + imgtype, bbox_inches = 'tight')
    plt.show()
    
    if is_discrete:
        dpms.calc_mod_mass_force(show_plot=True)
    else:
        dpms.calc_mod_mass_force_meso(show_plot=True)
    
    plt.savefig(fname + "_forces" + imgtype)
    plt.show()
    
    
