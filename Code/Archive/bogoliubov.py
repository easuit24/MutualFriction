import numpy as np
import matplotlib.pyplot as plt

#from vortexclean import GPETimeEv as gpe 
from gpesim_1d import GPETimeEv as gpe

class GPEBogoliubov(): 
    '''
    Generates a wavefunction with Bogoliubov noise for a single GPE wavefunction
    '''
    def __init__(self, L = 50,  dtcoef = 0.1, npoints = 2**9, dim = 2, numVort = 2, spawnType = 'pair', numImagSteps = 2000, numRealSteps = 0,  antiV = False, dist = 3, imp = False, impgpe = None): 
        # Simulation parameters 
        self.L = L 
        self.dtcoef = dtcoef

        self.npoints = npoints
        self.numImagSteps = numImagSteps
        self.numRealSteps = numRealSteps

        self.dim = dim 

        # vortex parameters 
        self.numVort = numVort 
        self.spawnType = spawnType
        self.antiV = antiV
        self.vdist = dist 

        # import parameters
        self.imp = imp


        #self.gpe_obj = gpe(L = self.L, dtcoef = self.dtcoef, npoints = self.npoints, dim = self.dim, numVort = self.numVort, spawnType = self.spawnType, numImagSteps=self.numImagSteps, numRealSteps=self.numRealSteps, antiV = self.antiV, dist= self.vdist, imp = False)
        
        if not imp: 
            self.gpe_obj = gpe(L = 50)
        else: 
            self.gpe_obj = impgpe

        self.psi_gpe = self.gpe_obj.psi
        self.xi = None
        self.ki = None
        self.setGrids() 
        # define Bogoliubov parameters
        self.uv_arr = np.zeros((len(self.xi[0]),2)) # define uk and vk, the Bogoliubov coefficients 
        self.alist = None 
        self.psiwig = None 
        self.setBogoCoefs() 
        self.setAlphas() 
        self.genNoise()


    def setGrids(self): 
        '''
        Set the x and p grids according to the GPE object created without Bogoliubov noise 
        '''
        self.xi = self.gpe_obj.xi
        self.ki = self.gpe_obj.ki
    
    def setBogoCoefs(self): 
        '''
        Set an array with Bogoliubov coefficients uk and vk related to the particle-hole properties of the mixture 
        '''
        for i in range(len(self.ki[0])):
            if self.ki[0][i] == 0:
                self.uv_arr[i] = [0,0]
            else: 
                e_frac = np.sqrt(self.ki[0][i]**2/(self.ki[0][i]**2 + 4))
                self.uv_arr[i] = [0.5 * (e_frac**0.5 + e_frac**(-0.5)),0.5 * (e_frac**0.5 - e_frac**(-0.5)) ]

    def setAlphas(self, plot = False): 
        '''
        Set the random noise for the wavefunction drawn from a Gaussian distribution function centered around 0 and variance of 1/2  
        '''
        self.alist = np.random.normal(0, 0.5, self.npoints) +1j*np.random.normal(0, 0.5, self.npoints) # draw from a normal distribution 

        if plot: 
            plt.figure() 
            plt.scatter(np.real(self.alist), np.imag(self.alist))
            plt.show() 
            print(np.std(np.abs(self.alist)**2))

    def genNoise(self, plot = False): 
        psinoise = np.zeros_like(self.xi[0], dtype = np.complex_)

        for i in range(len(psinoise)): 
            x = self.xi[0][i]
            psinoise[i] = 1/np.sqrt(self.L) *np.sum(self.alist*self.uv_arr[:,0]*np.exp(1j*self.ki[0]*x)+np.conj(self.alist)*self.uv_arr[:,1]*np.exp(-1j*self.ki[0]*x))

        if plot: 
            plt.figure() 
            plt.scatter(np.real(psinoise), np.imag(psinoise))
            plt.show() 

        self.psiwig = self.psi_gpe + psinoise

        # plt.figure() 
        # plt.plot(self.xi[0], self.psiwig)
        # plt.show()

   