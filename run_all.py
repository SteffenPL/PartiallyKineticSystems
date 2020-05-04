#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:03:02 2019

@author: plunder
"""

import numpy as np
from examples_monte_carlo_variance import analyse_variance



for name in ["linear"]: #, "onion", "singular"]:
    for sim_type in ["","_meso"]:
        runfile("example_" + name + sim_type + ".py")
        
    runfile("compare_disc_meso.py")
    
        
    n_particles = 2**np.arange(2,10)
    n_mc = 100
    analyse_variance(name, n_particles, n_mc)
    
    
