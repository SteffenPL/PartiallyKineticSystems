#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:30:55 2019

@author: plunder
"""

from save_state import basepath

comp_fname = basepath + "compare_" + pms.name

pms.plot_heavy_system()
dpms.plot_heavy_system()
plt.title("Macroscopic System")
plt.legend(["kinetic","discrete"], loc="lower center")
plt.savefig(comp_fname + "_heavy.png")
plt.savefig(comp_fname + "_heavy.eps")
plt.show()


dpms.plot_particle_paths(alpha=0.4)
pms.plot_particle_paths_meso_time()
plt.title("Cross-Bridges")
plt.savefig(comp_fname + "_particles_time.eps")
plt.savefig(comp_fname + "_particles_time.png")
plt.show()
