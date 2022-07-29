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

class Thermosphere():
    def __init__(self, N = 512, t_d = 0, t_0 = 200): 
        self.nAlts = N
        self.t_prime_boundary = 0
        self.t_boundary = 200
        self.alts = np.linspace(100, 500, num = self.nAlts)
        self.temp = self.init_temp(self.alts)
        
        return
    
    def init_temp(self, alt_in_km):
        temp_in_k = 200 + 600 * np.tanh( (alt_in_km - 100) / 100.0)
        return temp_in_k
    
    def calculateQeuv(self, mass, n_density, SZA, cross_section, plot = False, atom = "O2"):
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
        "Calculate Energies"
        euv_file = 'euv_37.csv'
        euv_info = read_euv_csv_file(euv_file)
        intensity_at_inf = EUVAC(euv_info['f74113'], euv_info['afac'], f107, f107a)
        wavelength = (euv_info['short'] + euv_info['long'])/2
        energies = convert_wavelength_to_joules(wavelength)
        
        "Calculate Scale Height"
        h = calc_scale_height(mass, self.alts, self.temp)
        
        "Calculate Mass Density"
        density = calc_hydrostatic(n_density, h, self.temp, self.alts)
        
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
        if(plot):
            plot_value_vs_alt(self.alts, Qeuv, atom + '_qeuv.png', 'Qeuv - ' + atom + ' (W/m3)')
        
        return(Qeuv)
    
    def solve(self, Qeuv):
        
        "System matrix and RHS term"
        "Diffusion term"
        Diff =(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))

        "Source term"
        A = (-1/4e-4)*Diff
        F = np.ones(N+1)*Qeuv
        
        "Boundary condition at x=0"
        A[0,:] = np.concatenate(([1],np.zeros(N)))
        F[0] = self.t_boundary

        A[N+1,:] = np.concatenate((np.zeros(N-1),[-1, 1]))
        F[N+1]=0
  
        "Solution of the linear system AU=F"
        u = np.linalg.solve(A,F)
        
        return u
    
    def run(self):
        "Set Solar Zenith Angle"
        SZA = 0.0
       
        'read in EUV file' 
        euv_file = 'euv_37.csv'
        euv_info = read_euv_csv_file(euv_file)

        "Compute Q_euv for O, O2, and N2"
        Q_euv_O = self.calculateQeuv(mass_o,n_o_bc, SZA, euv_info['ocross'], True, "O")
        Q_euv_O2 = self.calculateQeuv(mass_o2,n_o2_bc, SZA, euv_info['o2cross'], True, "O2")
        Q_euv_N2 = self.calculateQeuv(mass_n2,n_n2_bc, SZA, euv_info['n2cross'], True, "N2")
        
        return
        "Solve for Ion Temperatures"
        T_o = self.solve(Q_euv_O)
        T_o2 = self.solve(Q_euv_O2)
        T_n2 = self.solve(Q_euv_N2)
        
        return(T_o,T_o2,T_n2)
