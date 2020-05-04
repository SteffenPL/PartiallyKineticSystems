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



n = 250


# masses
m_r = 20.
m_q = 10. / n

# stiffness
kappa_r = 1.
kappa_q = 1.0 / n

# forces
U_r = lambda r, dr: 0.5 * kappa_r * r**2
T_r = lambda r, dr: 0.5 * m_r * dr**2

U_q = lambda q, dq: 0.5 * kappa_q * q**2
T_q = lambda q, dq: 0.5 * m_q * dq**2


# constraint
factor = 1.
# g = lambda r, q : factor*q/(1+r**2) - q**3
# g = lambda r, q : -factor * sp.sin(0.1*q)/(1+r**2) + 0.1*q
# g = lambda r, q : q**2 - 1/(1+(r-0.5)**2)   # start within the circle!
g = lambda r, q : q + r
# g = lambda r, q : q**2 + r**2  # looks nice and smooth!
# g = lambda r, q : 



# initial distribution
r0 = 1.
dr0 = 0
# Q0 = np.random.normal(loc=0.9, scale=1., size=(n,))

# generate grid
loc = 2.
scale = 1
qgrid_size = 100
qgrid = np.linspace(-5,7, qgrid_size)
rho0 = n * norm.pdf( qgrid, loc=loc, scale=scale )
t_end = 60


pms.init_equations(T_r, U_r, T_q, U_q, g)
pms.init_meso(r0, dr0, rho0, qgrid, t_end, n_eval=400)
pms.simulate_meso(method="Radau",atol=1.e-8,rtol=1.e-8,use_upwind=True)



pms.name = "linear"
save_plots( plt, pms, "meso", imgtype=".png")
save_plots( plt, pms, "meso", imgtype=".eps")
#
#
#
#
#
#
#
#
