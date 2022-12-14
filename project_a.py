# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:16:25 2022

@author: alexp
"""
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from helper_functions import *
from physics_functions import *
import matplotlib.animation as animation
from scipy.optimize import fsolve

# set f107:
f107 = 100.0
f107a = 100.0
efficiency = 0.3

# boundary conditions for densities:
n_o_bc = 5.0e17 # /m3
n_o2_bc = 4.0e18 # /m3
n_n2_bc = 1.7e19 # /m3


# Elemental Masses
mass_o = 16.0 # in AMU
mass_o2 = 32.0 # in AMU
mass_n2 = 28.0 # in AMU

"Calculate Energies"
euv_file = 'euv_37.csv'
euv_info = read_euv_csv_file(euv_file)
intensity_at_inf = EUVAC(euv_info['f74113'], euv_info['afac'], f107, f107a)
wavelength = (euv_info['short'] + euv_info['long'])/2
energies = convert_wavelength_to_joules(wavelength)

class Thermosphere():
    def __init__(self, N = 1024, t_d = 0, t_0 = 200): 
        self.nAlts = N
        self.t_prime_boundary = 0
        self.t_boundary = 200
        self.alts = np.linspace(100, 500, num = self.nAlts+2)
        self.mass = 0
        self.n = 0
        self.SZA = 0
        self.cross_section = 0
        self.species = ""
        self.A = 0
        
        return
    
    def init_temp(self, alt_in_km):
        temp_in_k = 200 + 600 * np.tanh( (alt_in_km - 100) / 100.0)
        return temp_in_k
    
    def calculateQeuv(self,T, mass, n_density, SZA, cross_section, plot = False, atom = "O2"):
        """
        Calculate neutral heating in the thermospheere.

        Parameters
        ----------
        mass : float
            mass of neutral.
        n_density : float
            number density of the neutral.
        SZA : degree
            Solar zenith angle.
        cross_section : float
            cross section of neutral atom

        Returns
        -------
        Qeuv

        """

        
        "Calculate Scale Height"
        h = calc_scale_height(mass, self.alts, T)
        
        "Calculate Mass Density"
        density = calc_hydrostatic(n_density, h, T, self.alts)
        
        "Calculate Optical Depth"
        tau = calc_tau(SZA, density, h, cross_section)
        
        "Calculate Q_euv"
        Qeuv = calculate_Qeuv(density,
                        intensity_at_inf,
                        tau,
                        cross_section,
                        energies,
                        efficiency)
        
        "Plot Results"
        #if(plot):
        #    plot_value_vs_alt(self.alts, Qeuv, atom + '_qeuv.png', 'Qeuv - ' + atom + ' (W/m3)')
        
        return(Qeuv)
    
    def solve(self, Qeuv, T, dt = 1):
        "Source term"
        #print(Qeuv*(4e-4))
        F = Qeuv/(9e-8)*np.ones(self.nAlts+2) + T/dt # temp
        #print(F)
        "Boundary condition at x=0"
        F[-1]=0; F[0] = 200
      
        "Solution of the linear system AU=F"
        u = np.linalg.solve(self.A,F)
        
        return(u)
    
    def run(self):
        "Set Solar Zenith Angle"
        SZA = 0.0
       
        'read in EUV file' 
        euv_file = 'euv_37.csv'
        euv_info = read_euv_csv_file(euv_file)

        "Time Array and Temperature"
        T0 = self.init_temp(self.alts)
        times = np.arange(0, 24*60*60, 5*60); dt = 300
        T = np.zeros((times.shape[0], T0.shape[0]))
        T[0] = T0

        "Diffusion term"
        Dx = (500-100)/self.nAlts
        Diff = (2*np.diag(np.ones(self.nAlts+2)) \
                         - np.diag(np.ones(self.nAlts+1),-1) - np.diag(np.ones(self.nAlts+1),1))
        A = (1/Dx**2)*Diff
        
        "Temporal Term for A"
        A = A + np.identity(A.shape[-1])/dt
        
        "Boundary Conditions"
        A[0,:] = np.concatenate(([1],np.zeros(self.nAlts+1)))
        A[-1,:] = np.concatenate((np.zeros(self.nAlts-1),[1/2, -2, 3/2]))/Dx
        self.A = A
        
 
        
        "Update Temperature Profile"
        for n in range(times.shape[0]-1):
            "Compute Q_euv for O, O2, and N2"
            Q_euv_O = self.calculateQeuv(T[n],mass_o,n_o_bc, SZA, euv_info['ocross'], False, "O")
            Q_euv_O2 = self.calculateQeuv(T[n],mass_o2,n_o2_bc, SZA, euv_info['o2cross'], False, "O2")
            Q_euv_N2 = self.calculateQeuv(T[n], mass_n2,n_n2_bc, SZA, euv_info['n2cross'], False, "N2")
            Q_euv = Q_euv_O + Q_euv_O2 + Q_euv_N2

            
 
            "Solve for Ion Temperatures"
            T[n+1] = self.solve(Q_euv, T[n], dt)
      
            #plt.clf()
        plt.plot(T[0], self.alts)
        plt.plot(T[len(T)//2], self.alts)
        plt.plot(T[-1], self.alts)
        

        
        return(T)
    
if(__name__ == '__main__'):
    th = Thermosphere()
    T = th.run()
    
    fig = plt.figure()
    ax = plt.axes() 
    myAnimation, = ax.plot([], [],':ob',linewidth=2)
    plt.grid()
    plt.xlabel("x",fontsize=16)
    plt.ylabel("u",fontsize=16)
    
    def animate(i):
        plt.plot(T[i],th.alts)
        myAnimation.set_data(T[i],th.alts)
        return myAnimation,

    anim = animation.FuncAnimation(fig,animate,frames=range(1,T.shape[0]),blit=True,repeat=True, interval = 10)
    
