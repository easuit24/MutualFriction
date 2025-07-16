
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, dst, dstn, idstn 
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
# import modules 
from vortexclean import GPETimeEv as gpev
from gpeboxclean import GPETimeEv as gpeb


class FiniteTempGPE():
    def __init__(self, L = 50, npoints = 2**9, dim = 2, numImagSteps = 500, numRealSteps = 1000, T = 0, Nsamples = 10, winMult = 2, dtcoef = 0.05, vortex = False, Tfact = 1/50, imp = False, impPsi = None, runAnim = False, animFileName = None, dst = True): 
        self.L = L 
        self.winMult = winMult
        self.winL = self.L*self.winMult
        self.npoints_input = npoints
        self.npoints = npoints *self.winMult
        
        self.dim = dim
        self.numImagSteps = numImagSteps
        self.numRealSteps = numRealSteps
        self.Tfact = Tfact
        self.T = T
        self.dtcoef = dtcoef
    
        self.Nsamples = Nsamples # number of classical noise distributions to generate 

        self.vortex = vortex 

        self.gpeobj = None
        self.gs = None 

        self.genGroundState() 


        # grid parameters to be initialized 
        self.dx = None
        self.dk = None 
    

        # noise parameters 
        self.psinoise = None 
        self.wf = None 
        self.classical_wavefunctions = None # for storing the wavefunctions for classical noise (real space)
        self.wf_samples = None
        self.average_thnoise = None 

        self.setGridParameters() 
        self.xi = self.setxgrid() 
        self.ki, self.dk, self.T = self.setpgrid()   

        self.imp = imp 
        

        self.time_tracking = [] 
        self.dst = dst

        if not imp: 
            self.simulate(self.numRealSteps)
        else: 
            self.impPsi = impPsi
            self.simulate(self.numRealSteps)

        self.runAnim = runAnim 
        self.animFileName = animFileName 

        if self.runAnim: 
           
            self.animatepsi2d(self.animFileName) 

        # generate the wavefunctions for the propagation 
        # self.genNoiseSamples()
        # self.assembleWavefunction() 

    def setGridParameters(self): 
        '''
        Define the grid spacing parameters 
        '''
        self.dx = self.gpeobj.dx 
        self.dk = self.gpeobj.dk

    def setxgrid(self): 
        '''
        Set up the x grid containing discrete values to evaluate the GPE on. The grid is evaluated over a window around the box potential 
        '''

        # set x grid 
        axes = [] 
        for i in range(self.dim): 

            axes.append(np.linspace(-self.winL/2,(self.npoints-1)*self.dx-self.winL/2,self.npoints) )
            
        axes_arr = np.array(axes) 
        xi = np.meshgrid(*axes_arr) 
        return xi 
        
    def setpgrid(self): 
        '''
        Set up the momentum (p) grid containing discrete momentum values corresponding to the spatial values defined in setxgrid() 
        '''
        paxes = [] 
        for i in range(self.dim): 
            #paxes.append(ifftshift(np.linspace(-np.pi/self.dx, (self.npoints - 1) * (2*np.pi)/self.winL - (np.pi/self.dx), self.npoints))[1:]) # do not count the zero momentum state 
            paxes.append(ifftshift(np.linspace(-np.pi/self.dx, (self.npoints - 1) * (2*np.pi)/self.winL - (np.pi/self.dx), self.npoints)))
        paxes_arr = np.array(paxes) 

        ki = np.meshgrid(*paxes_arr) 
        dk = 2*np.pi/(self.dx * self.npoints)
        kmax = np.pi / self.dx 

        T = kmax**2/2  # sets the temperature based on kmax 
        T = kmax**2/2*self.Tfact
        #T = 0
        #T = kmax**2/2/5

        return ki, dk, T 

    def genGroundState(self): 
        '''
        Generate a GPE object and initialize the ground state wavefunction psi from the result of imaginary time propagation. This model assumes a pair of vortices
        exists along the same x-axis. 
        '''
        if self.vortex: 
            self.gpeobj = gpev(L = self.L, dtcoef = self.dtcoef, dim = self.dim, numImagSteps=self.numImagSteps, runDyn = False, winMult=self.winMult)
        else: 

            self.gpeobj = gpeb(L = self.L, npoints = self.npoints_input, dtcoef = self.dtcoef, dim = self.dim, numImagSteps=self.numImagSteps, runDyn = False, winMult=self.winMult)
        self.gs = self.gpeobj.psi 

 

    # def assemble(self): 
    #     self.wf = self.gs + self.psinoise 


    # Generate an ensemble of psi noise wavefunctions to use in an average 

    def dst2d(self, wfk): 
        wfx_row = np.apply_along_axis(dst, axis = 1, arr = wfk, type = 2) 
        wfx = np.apply_along_axis(dst, axis = 0, arr = wfx_row, type = 2)
        return wfx 
    
    def genPsiK(self): 
        '''
        Generates the wavefunction psi randomly for a given k based on coefficients generated based on Gaussian random numbers. Assumes kb = m = 1
        '''
        #randr, randi = np.random.normal(size = 2) 
        randr = np.random.normal(size = (len(self.ki[0]), len(self.ki[0]))) 
        randi = np.random.normal(size = (len(self.ki[0]), len(self.ki[0]))) 

        
        #coef = np.sqrt(np.divide(2*self.T,(self.ki[0]**2 + self.ki[1]**2),out = np.zeros_like(self.ki[0]), where=(self.ki[0]**2 + self.ki[1]**2)!=0))
        coef = np.sqrt(np.divide(2*self.T,(self.ki[0]**2 + self.ki[1]**2),out = np.zeros_like(self.ki[0]), where=np.abs(self.ki[0]**2 + self.ki[1]**2)>1e-6))


        #return coef * (randr + 1j * randi)
        return np.multiply(coef, (randr + 1j * randi)/np.sqrt(2))
    
    def getNoise_dst2(self): 
        #coef=np.pi/self.winL
        coef = (2*self.L/self.dx**2)
        ksamples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        psix_arr = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        thermal_wf_samples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        for i in range(self.Nsamples): 
            ksamples[i] = self.genPsiK()
            #psik_sp = coef * (ksamples[i])
            innerbox = ifftshift(ksamples[i])[257:767, 257:767] # make this not hard coded later 
            psix_arr[i] = np.pad(idstn(coef*innerbox, type = 1), pad_width = (len(self.gs)-len(innerbox))//2, mode = 'constant', constant_values = 0)

            thermal_sample = self.gs + psix_arr[i] 
            thermal_sample[np.abs(self.gs)<0.1] = 0
            norm = np.sum(np.abs(thermal_sample)**2 * self.dx**2)

            thermal_wf_samples[i] = np.sqrt(self.gpeobj.Natoms/norm)*thermal_sample
        average_result = np.mean(np.abs(coef*ksamples)**2, axis = 0, dtype = np.complex_) # average of the noisy k samples 

        self.wf_samples = thermal_wf_samples 
        self.average_thnoise = average_result

    def getNoise_dst(self): 
        coef=2*self.winL/self.dx**2
        ksamples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        psix_arr = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        thermal_wf_samples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        for i in range(self.Nsamples): 
            ksamples[i] = self.genPsiK()
            psik_sp = coef * (ksamples[i])
            psix_arr[i] = idstn(psik_sp, type = 1) 

            thermal_sample = self.gs + psix_arr[i] 
            thermal_sample[np.abs(self.gs)<0.1] = 0
            norm = np.sum(np.abs(thermal_sample)**2 * self.dx**2)

            thermal_wf_samples[i] = np.sqrt(self.gpeobj.Natoms/norm)*thermal_sample
        average_result = np.mean(np.abs(coef*ksamples)**2, axis = 0, dtype = np.complex_) # average of the noisy k samples 

        self.wf_samples = thermal_wf_samples 
        self.average_thnoise = average_result
    
    def getNoise(self): 
        coef=self.winL/self.dx**2
        ksamples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        psix_arr = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        thermal_wf_samples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        for i in range(self.Nsamples): 
            ksamples[i] = self.genPsiK()
            psik_sp = coef * (ksamples[i])
            psix_arr[i] = ifft2(psik_sp) 
            thermal_sample = self.gs + psix_arr[i] 
            thermal_sample[np.abs(self.gs)<0.1] = 0
            norm = np.sum(np.abs(thermal_sample)**2 * self.dx**2)

            thermal_wf_samples[i] = np.sqrt(self.gpeobj.Natoms/norm)*thermal_sample
        average_result = np.mean(np.abs(coef*ksamples)**2, axis = 0, dtype = np.complex_) # average of the noisy k samples 

        self.wf_samples = thermal_wf_samples 
        self.average_thnoise = average_result 
        #return thermal_wf_samples, average_result 
    
    def realpropagate(self, wf, numSteps): 
        '''
        
        '''
        kinU = np.exp( -(1.0j )*(self.gpeobj.k2)*self.gpeobj.dt)
        
        snapshots = [wf] 
        #dynpsi = wf.copy() 

        if not self.imp: 
            dynpsi = wf.copy()
        else: 
            dynpsi = self.impPsi

        self.time_tracking = [0]

        for i in range(numSteps): 

            potU = np.exp(-(1.0j) *((self.gpeobj.Vbox)+self.gpeobj.g * np.abs(dynpsi)**2-1)*self.gpeobj.dt)

            psiFTold = fft2(dynpsi)
            psiFTnew = psiFTold * kinU 
            psiinterim = ifft2(psiFTnew)
            psinew = potU * psiinterim 
                
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            dynpsi = np.sqrt(self.gpeobj.Natoms)*psinew/np.sqrt(norm) 
                
            if (i%250 == 0):
                snapshots.append(dynpsi)
                self.time_tracking.append(self.gpeobj.dt * i)

        snapshots = np.array(snapshots)


        return snapshots, dynpsi
    
    def simulate(self, numSteps): 
        #self.getNoise() # initialize wavefunctions for samples 
        if self.dst: 
            self.getNoise_dst() 
        else: 
            self.getNoise()
        #self.getNoise()
        #self.snaps = np.array((self.Nsamples, len(self.wf_samples[0]),len(self.wf_samples[0])))
        self.final_psis = np.zeros((self.Nsamples, len(self.wf_samples[0]),len(self.wf_samples[0])), dtype = np.complex_)
        self.init_psis = self.wf_samples
        self.short_wfk = []
        for i in range(len(self.wf_samples)): 
            res = self.realpropagate(self.wf_samples[i], numSteps)
            self.snaps = res[0] 
            self.final_psis[i] = res[1]
            
            #self.gpeobj.simulatevortex()

            self.xgrid_short, self.psix_short, self.kgrid_short, self.psik_short = self.extractBox(self.xi, self.final_psis[i])
            self.short_wfk.append(self.psik_short)

        self.short_wfk = np.array(self.short_wfk)
        self.findAvgs()
        
        
    
    def extractBox(self, grid, wf): 
        '''
        Extract the inner box from the larger window of the simulation for plotting purposes 
        '''
        mask = (np.abs(grid[0])<self.L//2)&(np.abs(grid[1])<self.L//2)

        dim = int(np.sqrt(np.shape(grid[0][mask])))
        grid_clean = np.zeros((2,dim, dim))
        grid_clean[0] = grid[0][mask].reshape((dim, dim))
        grid_clean[1] = grid[1][mask].reshape((dim, dim))
        wf_clean = wf[mask].reshape(dim,dim)

        kwf_short = fft2(wf_clean)
        kgrid_short = self.ki[0][0][0:-2:self.winMult]

        return grid_clean, wf_clean, kgrid_short, kwf_short

    def fittingfunc(self, x, b, m): 
        return b* x**m

    def cleandata(self, grid, noise): 
        mask = (grid > 0) & (noise > 0)

        grid_clean = grid[mask]
        noise_clean = noise[mask]

        return grid_clean, noise_clean
    
    def findAvgs(self): 
        self.avg_dens = np.mean(np.abs(self.final_psis)**2, axis = 0, dtype = np.complex_)
        self.avg_kdens = np.mean(np.abs(self.short_wfk)**2, axis = 0, dtype = np.complex_)

    def animatepsi2d(self, filename):
        if filename != None: 
            path = fr"C:\Users\TQC User\Desktop\BECs\{filename}.mp4"

        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snaps[0])**2, extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot)
        
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
        time_text.set_text('time = 0')

        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        
        # plt.contour(self.xi[0], self.xi[1], self.Vs)
        
        plt.title(f'Animation for L={self.L}')
        fig.colorbar(data)

        def animate(i): 
            data.set_data(np.abs(self.snaps[i])**2)

            time_text.set_text('time = %.1d' % self.time_tracking[i])
            return data, time_text
        anim = animation.FuncAnimation(fig, animate, frames = len(self.snaps), blit = True)

        plt.show() 
        
        anim.save(path)

        return anim
        

    
    # def genProbK(self, k, psik, T, dk): 
    #     '''
    #     Generates the probability for the set of wavefunctions in k space based on the product of the individual probabilities 
    #     '''

    #     prob = np.exp(-np.abs(psik)**2 * (k[0][0][0]**2)/(2*T))
    #     for kval in range(len(k)-1):
    #         temp = np.exp(-np.abs(psik)**2 * ((k[0][0][kval+1])**2)/(2*T))
    #         prob = np.multiply(temp, prob) 
    #         #prob *= np.exp(-np.abs(psik)**2 * ((k[0][0][kval+1])**2)/(2*T))

    #     norm = np.sum(np.abs(prob)) * dk**2

    #     return prob / norm
    
    # def genNoiseSamples(self): 
    #     self.classical_wavefunctions = np.zeros((self.Nsamples, len(self.ki[0]), len(self.ki[0][0])), dtype = np.complex_)
    #     for i in range(self.Nsamples): 
    #         wf = self.genPsiK() # generate the wavefunction in k space 
    #         windowL = np.linspace(-self.winL/2,-self.L) 
    #         windowR = np.linspace(self.L, self.winL)
            
    #         self.classical_wavefunctions[i] = ifft2(wf) # shift to real space 

        

    # add the noise samples to the BEC ground state wavefunction 
    def assembleWavefunction(self): 
        self.wf_noisy = np.zeros_like(self.classical_wavefunctions)
        for i in range(self.Nsamples): 
            self.wf_noisy[i] = self.gs + self.classical_wavefunctions[i]

    # do real time evolution on the noisy wavefunctions 


    

