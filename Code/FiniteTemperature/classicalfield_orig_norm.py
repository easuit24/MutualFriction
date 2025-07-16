
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, dst, dstn, idstn 
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
# import modules 
from vortexclean import GPETimeEv as gpev
from gpeboxclean_orig import GPETimeEv as gpeb
from PointTracking_v2 import PointTracker as pt





class FiniteTempGPE():
    def __init__(self, L = 50, npoints = 2**9, dim = 2, numImagSteps = 500, numRealSteps = 1000, T = 0, boxthickness = 2, Nsamples = 10, winMult = 2, dtcoef = 0.05, vortex = False, antiV = False, Tfact = 1/50, imp = False, impPsi = None, runAnim = False, animFileName = None, dst = False): 
        self.L = L 
        self.winMult = winMult
        self.winL = self.L*self.winMult
        self.npoints_input = npoints
        self.npoints = npoints *self.winMult
        
        self.dim = dim
        self.numImagSteps = numImagSteps
        self.numRealSteps = numRealSteps
        self.Tfact = Tfact
        self.T = None 
        self.dtcoef = dtcoef

        self.boxthickness = boxthickness 
    
        self.Nsamples = Nsamples # number of classical noise distributions to generate 

        self.vortex = vortex 
        self.antiV = antiV

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
            if not self.vortex: 
                self.animatepsi2d(self.animFileName) 
            else: 
                self.animatepsi2d(self.animFileName)
                #self.animatepsi2d_vortex(self.animFileName)

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

        #T = kmax**2/2/50  # sets the temperature based on kmax 
        T = kmax**2/2*self.Tfact
        print("kmax: ", kmax)
        print("T: ", T)
        #T = 0
        #T = kmax**2/2/5

        return ki, dk, T 

    def genGroundState(self): 
        '''
        Generate a GPE object and initialize the ground state wavefunction psi from the result of imaginary time propagation. This model assumes a pair of vortices
        exists along the same x-axis. 
        '''
        if self.vortex: 
            self.gpeobj = gpev(L = self.L, npoints = self.npoints_input, boxthickness= self.boxthickness, dtcoef = self.dtcoef, dim = self.dim, numImagSteps=self.numImagSteps, antiV=self.antiV, runDyn = False, winMult=self.winMult)
        else: 

            self.gpeobj = gpeb(L = self.L, npoints = self.npoints_input, boxthickness= self.boxthickness, dtcoef = self.dtcoef, dim = self.dim, numImagSteps=self.numImagSteps, runDyn = False, winMult=self.winMult)
        self.gs = self.gpeobj.psi 
    
    ## TODO incorporate this into the class object with self. etc. 
    def cluster_vortices(self, vortex_positions, threshold = 1):
        '''
        
        '''
        if len(vortex_positions) == 0: 
            return [] 
        pos_array = np.array(vortex_positions) 
        clusters = [] 

        used_ind = set() 

        for i, pos in enumerate(pos_array): 
            if i in used_ind: 
                continue 
            cluster = [pos] 
            used_ind.add(i) 
        

            for j, other_pos in enumerate(pos_array): 
            
                if j in used_ind: 
                    continue 
                if (np.abs(pos[0] - other_pos[0])**2 + np.abs(pos[1] - other_pos[1])**2) < threshold:
                    cluster.append(other_pos) 
                    used_ind.add(j) 


            cluster_mean = np.mean(cluster, axis = 0) 
            clusters.append(tuple(cluster_mean))
        return clusters  

 

    def detect_vortices(self, psi, dx, L, previous_vortices=None, margin=10):
        '''
        Detect vortex positions by checking for 2π phase windings around plaquettes.
        Searches only within a region of interest around previous vortices.

        Parameters:
        psi - complex wavefunction (2D array)
        dx - grid spacing
        L - box length
        previous_vortices - list of (x, y) vortex positions in physical coordinates
        margin - number of grid points around previous vortex region to include

        Returns:
        vortex_positions - list of (x, y) tuples in physical coordinates
        '''
        psi = psi[int(L/2/dx):int(3*L/2/dx), int(L/2/dx):int(3*L/2/dx)] # keep the box only within the potential walls 
        Ny, Nx = psi.shape
        phase = np.angle(psi)
        vortex_positions = []

        # Convert physical positions to indices
        if previous_vortices is None or len(previous_vortices) == 0:
            # Search the entire domain if no previous positions are given
            i_min, i_max = 1, Ny - 2
            j_min, j_max = 1, Nx - 2
        else:
            i_indices = []
            j_indices = []
            for x, y in previous_vortices:
                j_indices.append(int(x / dx))
                i_indices.append(int(y / dx))

            i_min = max(1, min(i_indices) - margin)
            i_max = min(Ny - 2, max(i_indices) + margin)
            j_min = max(1, min(j_indices) - margin)
            j_max = min(Nx - 2, max(j_indices) + margin)

        # Main vortex detection loop
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                phi = [phase[i, j], phase[i, j+1], phase[i+1, j+1], phase[i+1, j]]
                dphi = np.diff(phi + [phi[0]])
                dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
                winding_number = np.sum(dphi) / (2 * np.pi)

                psi_dens = np.abs(psi)**2
                threshold = 0.0 # Adjust if needed
                
                if np.abs(winding_number) > 0.95 and np.min([psi_dens[i,j], psi_dens[i,j+1], psi_dens[i+1,j], psi_dens[i+1,j+1]]) > threshold:
                    #print([psi_dens[i,j], psi_dens[i,j+1], psi_dens[i+1,j], psi_dens[i+1,j+1]])
                    x = (j + 0.5) * dx
                    y = (i + 0.5) * dx
                    vortex_positions.append((x, y))
        # vortex_positions can become the next previous_vortices 
        #pos_arr = np.array(vortex_positions)

        clustered_positions = self.cluster_vortices(vortex_positions, threshold = 7*dx)
        #return vortex_positions
        return clustered_positions


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



    
    def getNoise(self): 
        coef=self.winL/self.dx**2
        #coef=self.L/self.dx**2
        ksamples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        psix_arr = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        thermal_wf_samples = np.zeros((self.Nsamples, len(self.ki[0][0]), len(self.ki[0][0])), dtype = np.complex_)
        self.N_th = np.zeros(self.Nsamples) 
        for i in range(self.Nsamples): 
            ksamples[i] = self.genPsiK()
            psik_sp = coef * (ksamples[i])
            psix_arr[i] = ifft2(psik_sp) 
            psix_arr[i][np.abs(self.gs)<0.1] = 0 
            ## Get number of atoms in the BEC 
            N_bec = self.gpeobj.Natoms - np.sum(np.abs(psix_arr[i])**2) 
            # Renormalize the BEC number of atoms
            bec_norm = np.sum(np.abs(self.gs)**2*self.dx**2) 
            self.bec_wf_renormalized = np.sqrt(N_bec/bec_norm) * self.gs 
            #thermal_sample = self.gs + psix_arr[i] 
            thermal_sample = self.bec_wf_renormalized + psix_arr[i]
            ##
            #thermal_sample[np.abs(self.gs)<0.1] = 0
            norm = np.sum(np.abs(thermal_sample)**2 * self.dx**2)
            print("Norm: ", norm)
            thermal_wf_samples[i] = np.sqrt(self.gpeobj.Natoms/norm)*thermal_sample
            # find the total number of thermal atoms 
            self.N_th[i] = np.sum(np.abs(ifft2(thermal_wf_samples[i]))**2)
        average_result = np.mean(np.abs(coef*ksamples)**2, axis = 0, dtype = np.complex_) # average of the noisy k samples 

        self.wf_samples = thermal_wf_samples 
        self.average_thnoise = average_result 
        #return thermal_wf_samples, average_result 
    
    def realpropagate(self, wf, numSteps): 
        '''
        
        '''
        #kinU = np.exp( -(1.0j )*(self.gpeobj.k2)*self.gpeobj.dt)
        kinU = np.exp( -(1.0j )*(self.gpeobj.k2)*self.gpeobj.dt)
        
        snapshots = [wf] 
        #dynpsi = wf.copy() 

        if not self.imp: 
            dynpsi = wf.copy()
        else: 
            dynpsi = self.impPsi

        self.time_tracking = [0]

        for i in range(numSteps): 
            #potU = np.exp(-(1.0j ) *((self.gpeobj.Vbox)+self.gpeobj.g * np.abs(dynpsi)**2-1)*self.gpeobj.dt)
            potU = np.exp(-(1.0j ) *((self.gpeobj.Vbox)+self.gpeobj.g * np.abs(dynpsi)**2-1)*self.gpeobj.dt)

            psiFTold = fft2(dynpsi)
            psiFTnew = psiFTold * kinU 
            psiinterim = ifft2(psiFTnew)
            psinew = potU * psiinterim 
                
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            dynpsi = np.sqrt(self.gpeobj.Natoms)*psinew/np.sqrt(norm) 
                
            #if (i%250 == 0):
            if (i%1000 == 0):
                snapshots.append(dynpsi)
                self.time_tracking.append(self.gpeobj.dt * i)

        snapshots = np.array(snapshots, dtype = np.complex64)


        return snapshots, dynpsi
    
    def simulate(self, numSteps): 
        #self.getNoise() # initialize wavefunctions for samples 

        self.getNoise()

        # TODO: Re-normalize the BEC number of atoms here - ensure the total number of atoms N_total = N_bec + N_th where Natoms = N_total

        #self.getNoise()
        #self.snaps = np.array((self.Nsamples, len(self.wf_samples[0]),len(self.wf_samples[0])))
        self.final_psis = np.zeros((self.Nsamples, len(self.wf_samples[0]),len(self.wf_samples[0])), dtype = np.complex64)
        self.init_psis = self.wf_samples
        self.short_wfk = []
        for i in range(len(self.wf_samples)): 
            #if not self.vortex: 
            self.N_bec = self.gpeobj.Natoms - self.N_th[i] 
            #TODO: Renormalize based on this value
            #norm = np.sum(np.abs(self.wf_samples[i])**2 * self.dx**2)
            #wf_samples_renormed = np.sqrt(self.N_th[i]/norm)*self.wf_samples[i]
            res = self.realpropagate(self.wf_samples[i], numSteps) 
            #res = self.realpropagate(wf_samples_renormed, numSteps) 

            self.snaps = res[0] 
            self.final_psis[i] = res[1]
            print(np.shape(self.snaps))
            print(np.shape(self.final_psis))
            # for vortex tracking 
            
            self.xgrid_short, self.psix_short, self.kgrid_short, self.psik_short = self.extractBox(self.xi, self.final_psis[i])
            self.short_wfk.append(self.psik_short)

            # if self.vortex: 
            #     prev_pos = None 
            #     vortex_positions = []
            #     for j in range(len(self.snaps)): 
            #         vortex_pos = self.detect_vortices(self.snaps[j], self.dx, self.L, prev_pos)
            #         prev_pos = vortex_pos  
            #         if len(prev_pos) == 2: 
            #             vortex_positions.append(np.array([vortex_pos[0][0], vortex_pos[0][1], vortex_pos[1][0], vortex_pos[1][1]])) 
            #         elif len(prev_pos) ==1:
            #             vortex_positions.append(np.array([vortex_pos[0][0], vortex_pos[0][1], np.nan, np.nan]))
            #             print("less than 2 vortices")
            #         elif len(prev_pos) > 2: 
            #             print('more than 2 vortices')
            #         elif len(prev_pos) == 0: 
            #             print('no vortices found')

            #     self.vortex_positions = np.array(vortex_positions)

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
            path = fr"C:\Users\TQC User\Desktop\BECs2\{filename}.mp4"

        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snaps[0])**2, extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot, origin = 'lower')
        # v1,v2 = None 
        # if self.vortex: 
        #     v1 = plt.scatter(self.vortex_positions[0][0]/self.dx+self.winL//4, self.vortex_positions[0][1]/self.dx+self.winL//4, color = 'blue', marker = '<', s = 20, alpha = 0.3)
        #     v2 = plt.scatter(self.vortex_positions[0][2]/self.dx+self.winL//4, self.vortex_positions[0][3]/self.dx+self.winL//4, color = 'blue', marker = '>', s = 20, alpha = 0.3)
        
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
        time_text.set_text('time = 0')

        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        
        # plt.contour(self.xi[0], self.xi[1], self.Vs)
        
        plt.title(f'Animation for L={self.L}')
        fig.colorbar(data)

        def animate(i): 
            data.set_data(np.abs(self.snaps[i])**2)
            # fix these: 
            # if self.vortex: 
            #     v1.set_offsets([self.vortex_positions[i][0]/self.dx+self.winL//4, self.vortex_positions[i][1]/self.dx+self.winL//4])
            #     v2.set_offsets([self.vortex_positions[i][2]/self.dx+self.winL//4, self.vortex_positions[i][3]/self.dx+self.winL//4])

            time_text.set_text('time = %.1d' % self.time_tracking[i])
            return data, time_text
            #return data, time_text, v1, v2 
        anim = animation.FuncAnimation(fig, animate, frames = len(self.snaps), blit = True)

        #plt.show() 
        
        anim.save(path)

        return anim
    
    
        
    def animatepsi2d_vortex(self, filename):
        if filename != None: 
            path = fr"C:\Users\TQC User\Desktop\BECs2\{filename}.mp4"

        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snaps[0])**2, extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot, origin = 'lower')

        if self.vortex: 
            v1 = plt.scatter(self.xi[0][0][int((self.vortex_positions[0][0]+1)/self.dx+self.winL/4/self.dx)], self.xi[0][0][int((self.vortex_positions[0][1]+1)/self.dx+self.winL/4/self.dx)], color = 'blue', marker = '<', s = 20, alpha = 0.6)
            v2 = plt.scatter(self.xi[0][0][int((self.vortex_positions[0][2]+1)/self.dx+self.winL/4/self.dx)], self.xi[0][0][int((self.vortex_positions[0][3]+1)/self.dx+self.winL/4/self.dx)], color = 'blue', marker = '>', s = 20, alpha = 0.6)
        
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
        time_text.set_text('time = 0')

        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        
        # plt.contour(self.xi[0], self.xi[1], self.Vs)
        
        plt.title(f'Animation for L={self.L}')
        fig.colorbar(data)

        def animate(i): 
            data.set_data(np.abs(self.snaps[i])**2)
            # fix these: 
            if self.vortex: 
                v1.set_offsets([self.xi[0][0][int((self.vortex_positions[i][0]+1)/self.dx+self.winL/4/self.dx)], self.xi[0][0][int((self.vortex_positions[i][1]+1)/self.dx+self.winL/4/self.dx)]])
                v2.set_offsets([self.xi[0][0][int((self.vortex_positions[i][2]+1)/self.dx+self.winL/4/self.dx)], self.xi[0][0][int((self.vortex_positions[i][3]+1)/self.dx+self.winL/4/self.dx)]])
            time_text.set_text('time = %.1d' % self.time_tracking[i])
            #return data, time_text
            return data, time_text, v1, v2
        anim = animation.FuncAnimation(fig, animate, frames = len(self.snaps), blit = True)

        plt.show() 
        
        anim.save(path)

        return anim

    

