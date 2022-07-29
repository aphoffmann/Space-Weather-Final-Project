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
        self.temp = self.init_temp(self.alts)
        self.mass = 0
        self.n = 0
        self.SZA = 0
        self.cross_section = 0
        self.species = ""
        
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
        Dx = (500-100)/self.nAlts
        Diff = (2*np.diag(np.ones(self.nAlts+2)) \
                         - np.diag(np.ones(self.nAlts+1),-1) - np.diag(np.ones(self.nAlts+1),1))
        Diff[0,:] = np.concatenate(([1],np.zeros(self.nAlts+1)))
        Diff[-1,:] = np.concatenate((np.zeros(self.nAlts),[-1, 1]))

            
        print("Diff: \n", Diff)
        "Source term"
        A = (1/Dx**2)*Diff
        F = Qeuv*(-4e-8)*np.ones(self.nAlts+2) # temp
        print("F: \n", F)
    
        
        "Boundary condition at x=0"
        #F[0] = self.t_boundary
        F[-1]=0
      
        'calulate inverse term'
        dt = 1
        T = np.linalg.inv(np.identity(A.shape[-1])-dt*A)@(dt*Qeuv + self.temp)
        
        "Solution of the linear system AU=F"
        u = np.linalg.solve(A,F)
        # print("U: \n", u)
        
        return(T)
    
    def run(self):
        "Set Solar Zenith Angle"
        SZA = 0.0
       
        'read in EUV file' 
        euv_file = 'euv_37.csv'
        euv_info = read_euv_csv_file(euv_file)

        "Compute Q_euv for O, O2, and N2"
        Q_euv_O = self.calculateQeuv(mass_o,n_o_bc, SZA, euv_info['ocross'], False, "O")
        Q_euv_O2 = self.calculateQeuv(mass_o2,n_o2_bc, SZA, euv_info['o2cross'], False, "O2")
        Q_euv_N2 = self.calculateQeuv(mass_n2,n_n2_bc, SZA, euv_info['n2cross'], False, "N2")
        Q_euv = Q_euv_O + Q_euv_O2 + Q_euv_N2
        
        "Solve for Ion Temperatures"
        T = self.solve(Q_euv)

        
        return(T)
    
    
