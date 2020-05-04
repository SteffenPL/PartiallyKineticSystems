#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:38:20 2019

@author: plunder
"""

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

import sympy as sp
from sympy.utilities import lambdify

from scipy.integrate import solve_ivp

import sdeint as si

from matplotlib import animation
# system dependent!
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'



from init_plot_settings import init_plot_settings
init_plot_settings(plt)

class SymbolicsPMS:
    
    def __init__(self, T_r, U_r, T_q, U_q, g):
        
        r_sym, dr_sym, q_sym = sp.symbols('r, \dot{r}, Q_j')
        
        g_sym = g(r_sym,q_sym)
        g = lambdify( (r_sym,q_sym), g_sym, "numpy" )
        
        
        Dpsi_sym = - sp.diff( g_sym, r_sym) / sp.diff( g_sym, q_sym )
        dq_sym = Dpsi_sym * dr_sym
        
        grr_sym = sp.diff( g_sym, r_sym , 2 )
        gq_sym = sp.diff( g_sym, q_sym )
        gqq_sym = sp.diff( gq_sym, q_sym)
        
        grq_sym = sp.diff( g_sym, q_sym, r_sym)
        
        
        kappa_sym = dr_sym * grr_sym * dr_sym + dq_sym * gqq_sym * dq_sym + 2 * dr_sym * grq_sym * dq_sym
        
        Dpsi  = lambdify((r_sym,q_sym), Dpsi_sym, "numpy")
        DpsiT = Dpsi
        kappa = lambdify( (r_sym,dr_sym,q_sym), kappa_sym, "numpy")
        
        G_q = lambdify( (r_sym, q_sym), gq_sym , "numpy")
        
        
        
        def forces_from_energies( T, U, q, dq ):
            H = lambda a,b: T(a,b) + U(a,b)
            L = lambda a,b: T(a,b) - U(a,b)
            
            F_sym = sp.diff( L(q,dq) , q ) - sp.diff( L(q,dq), dq , q ) * dq
            F = lambdify( (q,dq), F_sym, "numpy")
            
            M_sym = sp.diff( L(q,dq), dq , 2)
            M = lambdify( (q,dq), M_sym, "numpy")
            
            
            return F, M, H, L, F_sym, M_sym
        
        F_r, M_r, H_r, L_r, F_r_sym, M_r_sym = forces_from_energies(T_r, U_r, r_sym, dr_sym)
        F_q, M_q, H_q, L_q, F_q_sym, M_q_sym = forces_from_energies(T_q, U_q, q_sym, dr_sym)
        
        
        for key, var in locals().items():
            setattr(self, key, var)
        
        self.singular_manifolds = None
                        
        
        M_int = lambda r, dr, Q: Dpsi(r,Q) * M_q(Q,self.v_mod(r,dr,Q)) * Dpsi(r,Q) + 0.*Q
        F_int = lambda r, dr, Q: F_q(Q,self.v_mod(r,dr,Q)) + M_q(Q,self.v_mod(r,dr,Q)) / G_q(r,Q)  * kappa(r,dr,Q) + 0.*Q 
        
        self.v_mod = lambda r, dr , Q: Dpsi(r,Q) * dr + 0*Q
        
        self.M_mod = lambda r, dr, Q: M_r(r,dr) + sum( M_int(r,dr,Q) )
        self.F_mod = lambda r, dr, Q: F_r(r,dr) + sum( Dpsi(r,Q) * F_int(r,dr,Q) ) 
        self.F_mod_friction = lambda r, dr, Q, mu: F_r(r,dr) + sum( Dpsi(r,Q) * (F_int(r,dr,Q) - mu * self.v_mod(r,dr,Q)) ) 
        
        def build_row(row, nr, nq):
            z = zeros( (2*nr + nq, 2*nr + nq) )
            z[1,2:] = row
            return z
        
        self.G_noise = lambda r, dr, Q, G: build_row( Dpsi(r,Q) * G(Q), r.shape[0], Q.shape[0] )
    
            
        self.v_mod_meso = self.v_mod   # notice that it is expected to call v_mod_meso(r,dr,qgrid)

        # qx = q grid points,  qw = q quadrate weights
        self.M_mod_meso = lambda r, dr, rho, qx, qw: M_r(r,dr) + sum( M_int(r,dr,qx) * rho * qw )
        self.F_mod_meso = lambda r, dr, rho, qx, qw: F_r(r,dr) + sum( Dpsi(r,Q) * F_int(r,dr,qx) * rho * qw  ) 
        
        # not working
        self.F_mod_meso_noise = lambda r, dr, rho, qx, qw, G: M_r(r,dr) + sum( Noise_int(r,dr,qx, G) * rho * qw )
    
        
        
        
    def get_modified_eq(self):
        
        return self.M_mod, self.F_mod, self.v_mod
    
    def get_Dpsi(self):
        
        return self.Dpsi
        
    
    def compute_singular_points(self):
        
        if self.singular_manifolds is None:
            self.singular_manifolds = sp.solve( self.gq_sym, self.q_sym, self.r_sym )
        
        fncs = []
        
        for singular_manifold in self.singular_manifolds:
            fncs.append( lambdify( self.r_sym, singular_manifold[0] , "numpy") )
            
        return fncs
    
    def calc_energies(self, r, dr, Q):
        
        
        E_r_kin = self.T_r(r,dr)
        E_r_pot = self.U_r(r,dr)
        
        E_q_kin = np.sum( self.T_q(Q,self.v_mod(r,dr,Q)), axis=0 )
        E_q_pot = np.sum( self.U_q(Q,self.v_mod(r,dr,Q)), axis=0 )
        
        E_total = E_r_kin + E_r_pot + E_q_kin + E_q_pot

        return E_r_kin, E_r_pot, E_q_kin, E_q_pot, E_total    

    
    def calc_energies_meso(self, r, dr, rho, qx, qw):
        
        
        E_r_kin = self.T_r(r,dr)
        E_r_pot = self.U_r(r,dr)
        
        E_q_kin = np.sum( self.T_q(qx[:,np.newaxis],self.v_mod(r,dr,qx[:,np.newaxis])) * qw[:,np.newaxis] * rho, axis=0 )
        E_q_pot = np.sum( self.U_q(qx[:,np.newaxis],self.v_mod(r,dr,qx[:,np.newaxis])) * qw[:,np.newaxis] * rho, axis=0 )
        
        E_total = E_r_kin + E_r_pot + E_q_kin + E_q_pot

        return E_r_kin, E_r_pot, E_q_kin, E_q_pot, E_total      
    

class DiscretePMS:
    
    
    
    def __init__(self):
        
        self.nq = 1
        self.nr = 1
    
    
        self.col_min = np.array( [0.,0.4,0.2] )
        self.col_max = np.array( [0.6,0.9,0.0] )
    
        self.initial_eq = False
        self.initial_state = False
        
        
        
    def init_equations(self, T_r, U_r, T_q, U_q , g):
                
        self.pms_sym = SymbolicsPMS(T_r, U_r, T_q, U_q, g)


        self.sol = None
        self.initial_eq = True
        
        
    def init_state(self, r0, dr0 , Q0, t_end=10, n_eval=None):
        
        self.y0 = np.concatenate( [np.array([r0,dr0]), Q0] )
        self.t_end = t_end        
        
        self.t_end = t_end
        self.t_eval = None
        
        if n_eval is not None and n_eval > 0:
            self.t_eval = np.linspace(0, t_end, n_eval)
        
        self.n = len(Q0)
        
        self.sol = None
        self.initial_state = True
    
    
    def simulate(self, method="BDF", atol=1.e-5, rtol=1.e-6, G=None):
        
        if not self.initial_eq:
            print("Equations are not initialised.")
            return 
            
        if not self.initial_state:
            print("Initial conditions are missing.")
            return 
        
        # set up ode system
        def ode_fun(t, y):
            
            r = y[0]
            dr = y[1]
            Q = y[2:]
            
            M = self.pms_sym.M_mod(r,dr,Q)
            F = self.pms_sym.F_mod(r,dr,Q)
            v = self.pms_sym.v_mod(r,dr,Q)
            
            
            return np.concatenate( [np.array( [dr, F/M] ) , v ] )
        
        if G is not None:
            if self.t_eval is None:
                self.t_eval = np.linspace(0,self.t_end,1000)

            
            res = si.itoint( lambda y, t: ode_fun(t,y), G, self.y0, self.t_eval)
            
            self.sol = lambda:0
            self.sol.message = "SDE integrator finished."
            self.sol.y = res.T
            self.sol.t = self.t_eval
            
        else:
            self.sol = solve_ivp( ode_fun, [0,self.t_end], self.y0 , method=method, atol=atol, rtol=rtol, t_eval=self.t_eval)
        
        self.r  = self.sol.y[0,:]
        self.dr = self.sol.y[1,:]
        self.Q  = self.sol.y[2:,:]
        
        #print(self.sol.message)
        

    def init_meso(self, r0, dr0, rho0, qgrid, t_end=10, n_eval=None):
        
        self.y0_meso = np.concatenate( [np.array([r0,dr0]), rho0] )
        self.t_end = t_end        
        
        self.t_end = t_end
        self.t_eval = None
        
        if n_eval is not None and n_eval > 0:
            self.t_eval = np.linspace(0, t_end, n_eval)
        
        self.n = len(rho0)
        
        self.qgrid = qgrid
        self.dqgrid = qgrid[1:] - qgrid[0:-1]
        
        # weights for quadrature!
        self.weights = np.zeros_like(qgrid)
        self.weights[1:-1] = 0.5*( self.dqgrid[0:-1] + self.dqgrid[1:] )
        self.weights[0] = 0.5*qgrid[0]
        self.weights[-1] = 0.5*qgrid[-1]

        
        assert( np.any(np.array(self.dqgrid)>0) )
        
        self.sol = None
        self.initial_state = True
    
    
    def simulate_meso(self, method="BDF", atol=1.e-5, rtol=1.e-6, G=None):
        
        qgrid = self.qgrid
        dqgrid = self.dqgrid
        weights = self.weights
        
        inner = slice(1,-1)
        left = slice(0,-2)
        right = slice(2,None)
        
        # works inplace w.r.t. v!
        def upwind(v, rho, dqgrid):
            
            v_pos = 0.5*(v[inner] + np.abs(v[inner]))
            v_neg = 0.5*(v[inner] - np.abs(v[inner]))
            
            
            
            drho = np.zeros_like(rho)
            
            drho[inner] -= v_pos/dqgrid[0:-1]*(rho[inner] - rho[left])  # * (rho[inner] > 0.5)
            drho[inner] -= v_neg/dqgrid[1:]  *(rho[right] - rho[inner])  # * (rho[inner] > 0.5) 
        
            return drho
        
        def ode_fun_meso(t, y):
            
            r = y[0]
            dr = y[1]
            rho = y[2:]
            
            M = self.pms_sym.M_mod_meso(r, dr, rho, qgrid, weights)
            F = self.pms_sym.F_mod_meso(r, dr, rho, qgrid, weights)
            v = self.pms_sym.v_mod_meso(r, dr, qgrid)
            
            return np.concatenate( [np.array([dr, F/M]), upwind(v, rho, dqgrid)] )
        
        if G is not None:
            
            def G_func(y, t):
                r = y[0]
                dr = y[1]
                rho = y[2:]
                
                M = self.pms_sym.M_mod_meso(r, dr, rho, qgrid, weights)
                
                return np.concatenate( [np.array([0, G(y,t)/M]), upwind(v, rho, dqgrid)] )
                
            
            if self.t_eval is None:
                self.t_eval = np.linspace(0,self.t_end,1000)
            
            res = si.itoint( lambda y, t: ode_fun_meso(t,y), G_func, self.y0, self.t_eval)
            
            self.sol = lambda:0
            self.sol.message = "SDE integrator finished."
            self.sol.y = res.T
            self.sol.t = self.t_eval
            
        else:            
            self.sol = solve_ivp( ode_fun_meso, [0,self.t_end], self.y0_meso , method=method, atol=atol, rtol=rtol, t_eval=self.t_eval)
        
        self.r  = self.sol.y[0,:]
        self.dr = self.sol.y[1,:]
        self.rho  = self.sol.y[2:,:]
        
        #print(self.sol.message)
    
        
        
    def calc_domain(self, margin_factor=1.2):
         
        rmin = np.min(self.r)
        rmax = np.max(self.r)
        
        if hasattr(self,"Q"):
            qmin = np.min(self.Q)
            qmax = np.max(self.Q)
        else:
            qmin = np.inf
            qmax = -np.inf
            
        if hasattr(self,"qgrid"):
            qmin = min(qmin, self.qgrid[0])
            qmax = max(qmax, self.qgrid[-1])
        
        def zoom(a,b,f):
            m = 0.5*(a+b)
            w = b - m
            return m + f*w, m - f*w
        
        rmin, rmax = zoom(rmin,rmax,margin_factor)
        qmin, qmax = zoom(qmin,qmax,margin_factor)
        
        return rmin, rmax, qmin, qmax
    
    def plot_singularities(self, rmin=None, rmax=None, detail=100):
        if rmin is None or rmax is None:
            domain = self.calc_domain()

        singular_fncs = self.pms_sym.compute_singular_points()
        
        x_values = np.linspace(domain[0],domain[1],detail)
        for fnc in singular_fncs:
            plt.plot(x_values, fnc(x_values) + 0*x_values, 'r' )
        
        
    def plot_g(self, rmax=None, qmax=None, levels=10, detail=100, lw=1, alpha=1.):
        
        if rmax is None or qmax is None:
            rmin, rmax, qmin, qmax = self.calc_domain(1.2)
        
        r_grid, q_grid = np.mgrid[ rmin:rmax:(rmax-rmin)/detail , qmin:qmax:(qmax-qmin)/detail]
        plt.contour(r_grid, q_grid, self.pms_sym.g(r_grid,q_grid), levels=levels, alpha=alpha, linewidths=lw)
        
        
        plt.xlabel('r'), plt.ylabel('q'), plt.title('Constraint contours')
        
        
    def plot_g_img(self, rmax=None, qmax=None, detail=100, alpha=1.):
        
        if rmax is None or qmax is None:
            rmin, rmax, qmin, qmax = self.calc_domain(1.2)
        
        q_grid, r_grid = np.mgrid[ qmax:qmin:(-qmax+qmin)/detail, rmax:rmin:(-rmax+rmin)/detail ]
        
        Z = self.pms_sym.g(r_grid,q_grid)
        
        plt.imshow(Z, alpha=alpha, extent=(rmax,rmin,qmax,qmin), origin = 'lower', aspect=(rmax-rmin)/(qmax-qmin)*6/8)
        plt.axes
        
        plt.xlabel('r'), plt.ylabel('q'), plt.title('g(r,q)')
        
        
    def plot_veff_img(self, rmax=None, qmax=None, detail=100, alpha=1.):
        
        if rmax is None or qmax is None:
            rmin, rmax, qmin, qmax = self.calc_domain(1.2)
        
        q_grid, r_grid = np.mgrid[ qmax:qmin:(-qmax+qmin)/detail, rmax:rmin:(-rmax+rmin)/detail ]
        
        Z = self.pms_sym.v_mod(r_grid,1.,q_grid)
        
        plt.imshow(Z, alpha=alpha, extent=(rmax,rmin,qmax,qmin), origin = 'lower', aspect=(rmax-rmin)/(qmax-qmin)*6/8)
        plt.axes
        
        plt.xlabel('r'), plt.ylabel('q'), plt.title(r'$v_{\mathrm{eff}}(r,\dot{r} = 1, q)$')
    
    def plot_Q(self, r, Q):
        plt.plot( 1+ Q*0, Q, marker='.', lw=0 )
        plt.xlabel('r')
        plt.ylabel('q')
        
        

    def plot_heavy_system(self):
        plt.plot(self.sol.t, self.sol.y[0,:] )
        plt.xlabel("t"), plt.ylabel("r"), plt.title("heavy system")
        

    def plot_particle_paths(self, nParticles = 500, use_r_axis=False, plot_singular_pts=False, lw=0.3, alpha=1.):
        
        
        if use_r_axis:
            x_values = self.r
        else:
            x_values = self.sol.t
        
        if plot_singular_pts:
                
            singular_fncs = self.pms_sym.compute_singular_points()

            for fnc in singular_fncs:
                plt.plot(x_values, fnc(self.r) + 0*x_values, 'r' )
                
        
        step = int(self.n/nParticles) + 1
        for i in range(0, self.n, step):
            p = i/self.n
                
            plt.plot(x_values, self.Q[i,:], lw=lw, color = p*self.col_min + (1-p)*self.col_max, alpha=alpha )
        
        plt.ylabel("q"), plt.title("particles")

        if use_r_axis:
            plt.xlabel("r")
        else:
            plt.xlabel("t")
    
        
        
    def plot_particle_paths_meso_time(self, plot_singular_pts=False):
        
        rho = self.rho
        t = self.sol.t

        qmin = self.qgrid[0]
        qmax = self.qgrid[-1]
        
        plt.imshow( rho, extent=[0,self.t_end,qmin,qmax], cmap='YlGn', origin = 'lower', aspect=(self.t_end)/(qmax-qmin)*6/8)
        
        if plot_singular_pts:
                
            singular_fncs = self.pms_sym.compute_singular_points()

            for fnc in singular_fncs:
                plt.plot(t, fnc(self.r) + 0*t, 'r' )
                
        
                
        plt.xlabel("t"), plt.ylabel("q"), plt.title("particles")

            
    def plot_energies(self, energies, energy_colors):
    
        def plot_energy(k):
            plt.plot(self.sol.t, energies[k], color=energy_colors[k])

        plt.subplot(221)
        plot_energy(0)
        plot_energy(1)
        plt.xlabel("t"), plt.ylabel("Energy")
        plt.title("Heavy system")
        
        
        plt.subplot(222)
        plot_energy(2)
        plot_energy(3)
        plt.xlabel("t"), plt.ylabel("Energy")
        plt.title("Particle system")
        
        
        plt.subplot(212)
        for k in range(5):
            plot_energy(k)
        plt.xlabel("t"), plt.ylabel("Energy")
        plt.title("Coupled system")
        
        plt.legend([r'$T_r$',r'$U_r$',r'$T_Q$',r'$U_Q$',r"$E_{total}$"],loc='lower right')

    
        plt.tight_layout()
    
    def calc_energies(self, show_plot=False):
        
        r = self.r
        dr = self.dr
        Q = self.Q
        
        energies = self.pms_sym.calc_energies(r,dr,Q)
        
        energy_colors = ([0.,0.,1.],
                         [0.,0.,0.5],
                          [1. ,0.,0.],
                          [0.5,0.,0.],
                          [0. ,0.7,0.])
    
    
        if show_plot:
            self.plot_energies(energies, energy_colors)
        
        
        
    def calc_energies_meso(self, show_plot=False):
        
        r = self.r
        dr = self.dr
        rho = self.rho
        
        energies = self.pms_sym.calc_energies_meso(r,dr,rho,self.qgrid,self.weights)
        
        energy_colors = ([0.,0.,1.],
                         [0.,0.,0.5],
                          [1. ,0.,0.],
                          [0.5,0.,0.],
                          [0. ,0.7,0.])
    
    
        if show_plot:
            self.plot_energies(energies, energy_colors)
        
        
            
    def calc_mod_mass_force(self, show_plot=False):
        
        r = self.r
        dr = self.dr
        Q = self.Q
        t = self.sol.t
        
        pms = self.pms_sym
        
        M_mod_hist = pms.M_mod(r, dr, Q)
        F_mod_hist = pms.F_mod(r, dr, Q)
    
        if show_plot:
            plt.subplot(211)
            plt.plot(t, M_mod_hist)
            plt.plot(t, pms.M_r(r,dr) + 0*r , ":")
            plt.xlabel("t"), plt.ylabel("$M_{mod}$")
        
            plt.subplot(223)
            plt.plot(t, F_mod_hist)
            plt.plot(t, pms.F_r(r,dr) + 0*r , ":")
            plt.xlabel("t"), plt.ylabel("$F_{mod}$")
            
            
            plt.subplot(224)
            plt.plot(t, F_mod_hist / M_mod_hist, lw=2)
            plt.plot(t, F_mod_hist, ":")
            plt.plot(t, F_mod_hist / M_mod_hist[0], "r--")
            plt.xlabel("t"), plt.ylabel("$F_{mod} / M_{mod}$")
            
        plt.tight_layout()
        
        
    def calc_mod_mass_force_meso(self, show_plot=False):
        
        r = self.r
        dr = self.dr
        rho = self.rho
        t = self.sol.t
        
        qgrid = self.qgrid
        dqgrid = self.dqgrid
        weights = self.weights
        
        pms = self.pms_sym
        
        M_mod_hist = pms.M_mod_meso(r, dr, rho, qgrid[:,np.newaxis], weights[:,np.newaxis])
        F_mod_hist = pms.F_mod_meso(r, dr, rho, qgrid[:,np.newaxis], weights[:,np.newaxis])
    
        if show_plot:
            plt.subplot(211)
            plt.plot(t, M_mod_hist)
            plt.plot(t, pms.M_r(r,dr) + 0*r , ":")
            plt.xlabel("t"), plt.ylabel("$M_{mod}$")
        
            plt.subplot(223)
            plt.plot(t, F_mod_hist)
            plt.plot(t, pms.F_r(r,dr) + 0*r , ":")
            plt.xlabel("t"), plt.ylabel("$F_{mod}$")
            
            
            plt.subplot(224)
            plt.plot(t, F_mod_hist / M_mod_hist, lw=2)
            plt.plot(t, F_mod_hist, ":")
            plt.plot(t, F_mod_hist / M_mod_hist[0], "r--")
            plt.xlabel("t"), plt.ylabel("$F_{mod} / M_{mod}$")
            
        plt.tight_layout()
    
    def create_animation(sol, fname = None):
        
        
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes()
        
        plot_g(levels=100)
        
        pts, = ax.plot([],[],marker='.',lw=0)
        
        # initialization function: plot the background of each frame
        def init():
            pts.set_data([], [])
            return pts,
        
        # animation function.  This is called sequentially
        def animate(i):
            r = self.r[i]
            Q = self.Q[i]
            
            pts.set_data(r + 0*Q, Q)
            return pts,
        
        
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=sol.t.shape[0], interval=20, blit=True)
        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        anim.save(fname + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        
            