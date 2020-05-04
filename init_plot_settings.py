#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:24:51 2019

@author: plunder

This files just sets up the plot env.
"""


def init_plot_settings(plt, font_size=24, fig_size=(8,6), dpi=100):
    plt.rc( 'figure', figsize=fig_size ,  dpi=dpi)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=font_size)
    plt.rc('text.latex' , preamble = r'\usepackage{amsmath}')
    
    plt.rc('image', cmap='coolwarm')
    #plt.rcParams.update(params)
    