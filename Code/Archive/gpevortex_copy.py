import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from scipy import constants 
from scipy.fft import fft2, ifft2, fftshift, ifftshift 
import time 
import os 

class GPETimeEv():
 
    def __init__(self, L = 20, dtcoef = 0.1, npoints = 2**9, dim = 1, numVort = 10, spawnType = 'pair', tol = 10e-10, Nfactor = 1000, numImagSteps = 2000, numRealSteps = 10000, antiV = False, dist = 2, runDyn = True, imp = False, impPsi = None): 
        # define constants: assume we use Rubidum-87
        self.HBAR = constants.value(u'Planck constant') 
        self.SCAT = 90 * 0.0529e-9
        #self.MASS = 1.44e-25
        self.MASS = 84.911789732 * constants.value(u'atomic mass constant')
        
        self.L = L 
        self.winMult = 2 # multiplier: how much bigger the window is than the condensate
        self.winL = self.winMult*self.L # set the window length for the simulation
        self.npoints = self.winMult*npoints
        self.dx = self.winL/self.npoints 
        self.dk = None
        
        self.tol = tol
        self.imp = imp 
        
        self.dim = dim 
        self.xi = None
        self.ki = None
        self.psi_init = None
        self.psi = None
        self.dynpsi = None 
        self.snapshots = None
        self.k2 = 0
        self.Vpert = None
        self.Vbox = None 
        self.Vs = None
        self.setxgrid() 
        self.setpgrid()
        self.setk2()
        self.Natoms = Nfactor*npoints # set the number of atoms
        self.numVort = numVort # number of vortices to generate 

        self.g = self.L**2/self.Natoms

        self.g_sho = None
        self.dtcoef = dtcoef 
        self.dt = self.dtcoef*(self.dx**2) # coef was originally 0.1
        
        self.Vbox = self.setbox() 

        self.numImagSteps = numImagSteps
        self.numRealSteps = numRealSteps
        self.spawnType = spawnType 
        self.antiV = antiV
        self.dist = dist 
        self.runDyn = runDyn


        self.Ep_arr = []
        self.Ek_arr = [] 
        self.Ei_arr = [] 
        self.virial = [] 
        self.time_tracking = [] 


        #self.initpsi()
        
        # run the program 
        if not self.imp: 
            self.initpsi()
            self.simulateimag()
            #if self.runDyn == True: 
                #self.simulatevortex()
        
        else: 
            self.impPsi = impPsi 
            self.simulatevortex() 

        
        
    def setxgrid(self): 
        '''
        Set up the x grid containing discrete values to evaluate the GPE on 
        '''
    
        # set x grid 
        axes = [] 
        for i in range(self.dim): 
            axes.append(np.linspace(-self.winL/2,(self.npoints-1)*self.dx-self.winL/2,self.npoints) )
            
        axes_arr = np.array(axes) 
        self.xi = np.meshgrid(*axes_arr) 
        
    def setpgrid(self): 
        '''
        Set up the momentum (p) grid containing discrete momentum values corresponding to the spatial values defined above
        '''
        paxes = [] 
        for i in range(self.dim): 
            paxes.append(ifftshift(np.linspace(-np.pi/self.dx, (self.npoints - 1) * (2*np.pi)/self.winL - (np.pi/self.dx), self.npoints)))
        paxes_arr = np.array(paxes) 
        self.ki = np.meshgrid(*paxes_arr) 
        self.dk = 2*np.pi/(self.dx * self.npoints)
        
        
    def setbox(self):
        '''
        Set up the potential energy box to contain the wavefunction assuming a perfect box trap 
        '''
        V = 0
        for i in range(self.dim):
            V += 5 * np.array(np.power(np.e, -2*(self.xi[i] - np.ones_like(self.xi[i])*self.L/2)**2) + np.power(np.e, -2*(self.xi[i] + np.ones_like(self.xi[i])*self.L/2)**2))

        V[V<self.tol] = 0.0 # equivalent of Mathematica's Chop
        return V 
        

    def initpsi(self): 
        psi0 = np.ones_like(self.xi[0])
        
        psi0[np.abs(self.xi[0]) > self.L/2] = 0 
        psi0[np.abs(self.xi[1]) > self.L/2] = 0

        if self.spawnType == 'nonrand': 
            print('Generating non-random vortices...')
            psi0 = self.vortexFunc(psi0) 
        elif self.spawnType == 'rand': 
            print("Generating random vortices...")
            psi0 = self.vortexRand(psi0, prop = 0.5)
            psi0 = self.vortexRand(psi0, antiV = True, prop = 0.5) 
        elif self.spawnType =='pair': 
            psi0 = self.vortexPair(psi0, antiV = self.antiV)
        norm = np.sum(np.abs(psi0)**2) * self.dx**self.dim

         
        self.psi_init = np.sqrt(self.Natoms/norm) * psi0
        self.psi = np.sqrt(self.Natoms/norm) * psi0
        

        
    def setk2(self): 
        '''
        Defines the kinetic energy squared based on the values from the momentum grid 
        '''
        self.k2 = 0 
        for i in range(self.dim):
            self.k2 += (self.ki[i]**2)/2

    # def vortexPair2(self, psi, antiV = False): 

    #     r_grid = np.sqrt(self.xi[0]**2 + self.xi[1]**2)
    #     theta_grid = np.arctan2(self.xi[1], self.xi[0])

    #     x_loc, y_loc = self.dist, 0

    #     if antiV == False:        
    #         theta_grid += np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0]))
    #     else: 
    #         theta_grid -= np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0]))

    #     theta_grid += np.arctan2((self.xi[1]),(self.xi[0]))

    #     theta_grid -= np.arctan2((self.xi[1]),(self.xi[0]))
    #     psiVort = psi * np.tanh(r_grid)*np.exp(theta_grid * 1j) # modify the r grid too with the new points - make sure that the tanh is 0 at EACH vortex 
    #     # right now it is only 0 at the (0,0) coordinate always - modify r_grid to make this condition true at all vortices 
    #     #psiVort = psi * np.exp(theta_grid * 1j)
        
    #     #psiVort = psi *np.exp(theta_grid * 1j)
    #     return psiVort

    def vortexPair(self, psi, antiV = False): 
        '''
        Spawns a pair of vortices at a specified distance from one another. One is placed at the origin and the other is placed along the positive x axis
        at the specified distance. 

        Parameters: 
        psi - the GPE wavefunction (arr) 
        antiV (optional) - whether the vortex pair includes an anti-vortex (bool)

        Returns: 
        psiVort - New wavefunction including vortices  
        '''
        x_loc, y_loc = self.dist, 0 # set location of non-origin vortex 

        # set the radial grid including divets at each vortex 
        r_grid = np.sqrt(self.xi[0]**2 + self.xi[1]**2) 
        r_grid += np.sqrt((x_loc - self.xi[0])**2 + (y_loc - self.xi[1])**2)
        self.r_grid = r_grid 

        theta_grid = np.arctan2(self.xi[1], self.xi[0]) # set the phase for the origin vortex  

        # set the phase of the other vortex depending on if it is a vortex or anti-vortex 
        if antiV == False:        
            theta_grid += np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0])) # vortex/vortex  
            
        else: 
            theta_grid -= np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0])) # vortex/anti vortex 

        psiVort = psi * np.tanh(r_grid)*np.exp(theta_grid * 1j) 

        return psiVort
    
    def vortexFunc(self, psi, locs = [(1,0),(0,7),(0,0)], antiV = False): 
        '''
        Sets an array of vortices or anti-vortices at set locations in the box

        Parameters: 
        psi - the GPE wavefunction 
        locs (optional) - the locations of vortices to be spawned 
        antiV (optional) - whether the vortices are antivortices 

        Returns: 
        psiVort - New wavefunction including vortices 
        '''

        #r_grid = np.sqrt(self.xi[0]**2 + self.xi[1]**2)
        #theta_grid = np.arctan2(self.xi[1], self.xi[0]) 
        for v in locs: 
            x_loc, y_loc = v 
            print("Spawning Vortex at: ", x_loc, ",", y_loc)

            r_grid += np.sqrt((x_loc-self.xi[0])**2 + (y_loc-self.xi[1])**2) # set the vortex to appear at v

            if not antiV: 
                theta_grid += np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0])) # set the phase at the vortex at v assuming vortex 
            else: 
                theta_grid -= np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0])) # set the phase at the vortex at v assuming antivortex 

        #theta_grid -= np.arctan2((self.xi[1]),(self.xi[0]))
        psiVort = psi * np.tanh(r_grid)*np.exp(theta_grid * 1j)
        
        return psiVort
    
    def vortexRand(self, psi, antiV = False, prop = 1):
        '''
        Generates numVort number of vortices in the grid 

        Parameters: 
        psi - the GPE wavefunction 
        antiV (optional) - whether to spawn antivortices or vortices (bool)
        prop - the proportion of total vortices to spawn 

        Returns: 
        psiVort - the new wavefunction including vortices 
        '''
        #r_grid = np.sqrt(self.xi[0]**2 + self.xi[1]**2)
        #theta_grid = np.arctan2(self.xi[1], self.xi[0])
        x_loc, y_loc = np.random.uniform(-self.L/2,self.L/2), np.random.uniform(-self.L/2,self.L/2)
        r_grid = np.sqrt((x_loc-self.xi[0])**2 + (y_loc-self.xi[1])**2)
        if antiV == False: # for vortices 
            for v in range(int(prop*self.numVort)): 
                theta_grid += np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0]))

            #psiVort = psi * np.tanh(r_grid)*np.exp(theta_grid * 1j)
        else: # for antivortices 
            for v in range(int(prop*self.numVort)): 
                theta_grid -= np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0]))

        psiVort = psi * np.tanh(r_grid)*np.exp(theta_grid * 1j)
        return psiVort

    def calcEnergy(self): 
        '''
        Calculates the potential, kinetic, and interaction energy. The kinetic energy is calculated by 
        transforming to momentum space via a Fourier transform. The calculations are unitful. 
        '''
        dF = self.dx**2 / (np.sqrt(2 * np.pi) **2 )
        psik = np.fft.fftshift(np.fft.fftn(self.dynpsi * dF)) # transform to momentum space
        Ep = np.trapz(np.trapz(np.conj(self.dynpsi) * self.Vbox * self.dynpsi))*self.dx**2 
        Ek = np.trapz(np.trapz(np.conj(psik) * self.k2 * psik))*self.dk**2 # calculate KE in momentum space
        Ei = 0.5*self.g * np.trapz(np.trapz(np.abs(self.dynpsi)**4))*self.dx**2

        self.Ep_arr.append(Ep)
        self.Ek_arr.append(Ek) 
        self.Ei_arr.append(Ei) 
        


    def simulateimag(self): 
        '''
        Propagates the wavefunction in imaginary time. Starts from the initial tanh solution guess and uses
        the split step method (Suziki-Trotter expansion) to determine the resulting wavefunction. Reassigns
        psi to the result of this propagation process. 
        '''
        self.interimpsi = self.psi.copy()     
       #kinUim = np.power(np.e, -self.k2*self.dt)
        kinUim = np.exp(-self.k2*self.dt) 
        
        # set up the potential energy part with propagation loop 
        #print("Num Steps: ", self.numImagSteps)
        for i in range(self.numImagSteps): 

            #potUim = np.power(np.e, -(self.Vbox + self.g  * np.abs(self.psi)**2 - 1) *self.dt)
            t0 = time.time() 
            potUim = np.exp(-(self.Vbox + self.g  * np.abs(self.psi)**2 - 1) *self.dt)
            t1 = time.time() 
           # print(t1-t0) 
            psiFTold = fft2(self.psi) 
            t2 = time.time() 
           # print(t2-t1)
            # fix this to make nD later 
           # psiFTnew = np.array(psiFTold * kinUim) 
            psiFTnew = psiFTold * kinUim
            t3 = time.time() 
           # print(t3-t2)
            psiinterim = ifft2(psiFTnew)
            t4 = time.time() 
           # print(t4-t3)
            psinew = potUim * psiinterim 
            t5 = time.time() 
            #print(t5-t4)
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            t6 = time.time() 
           # print(t6-t5)
            self.psi = np.sqrt(self.Natoms/norm) * psinew
            t7 = time.time() 
           # print(t7-t6)
           # print("Total: ", t7-t0)



 

    def simulatevortex(self):
        '''
        Propagates the wavefunction in real time assuming the system has stablized
        after being confined in an optical box. The potential is instantaneously
        switched to a harmonic potential and the condensate is allowed to evolve.  
        '''
       # kinU = np.power(np.e, -(1.0j )*(self.k2)*self.dt)
        kinU = np.exp( -(1.0j )*(self.k2)*self.dt)
        
        self.snapshots = [self.psi] 
        if not self.imp: 
            self.dynpsi = self.psi.copy()
        else: 
            self.dynpsi = self.impPsi # set the dynamic wavefunction to be the imported one 
        
        for i in range(self.numRealSteps): 

           # potU = np.power(np.e, -(1.0j) *((self.Vbox)+self.g * np.abs(self.dynpsi)**2-1)*self.dt)
            potU = np.exp(-(1.0j) *((self.Vbox)+self.g * np.abs(self.dynpsi)**2-1)*self.dt)

            #psiFTold = np.fft.fftshift(np.fft.fft2(self.dynpsi))
            psiFTold = fft2(self.dynpsi)
            psiFTnew = psiFTold * kinU 
            #psiinterim = np.fft.ifft2(np.fft.ifftshift(psiFTnew))
            psiinterim = ifft2(psiFTnew)
            psinew = potU * psiinterim 

            
            # for j in range(len(psinew)-1):
                # norm += (np.abs(psinew[j])**2 + np.abs(psinew[j+1])**2)/2 * self.dx ** self.dim
                
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            self.dynpsi = np.sqrt(self.Natoms/norm) * psinew
             

            if (i%250 == 0):
                self.snapshots.append(self.dynpsi)
                self.calcEnergy() # calculates potential energy and add to array
                self.time_tracking.append(self.dt * i)

        self.snapshots = np.array(self.snapshots)
        self.time_tracking = np.array(self.time_tracking) 

    def animatepsi(self):
    
        '''
        Animates the snapshots in a time evolving graphic 
        
        Returns: 
        theAnim - the animation 
        '''

        fig=plt.figure()
        data, = plt.plot(self.xi[0], np.abs(self.snapshots[0]))
        plt.ylim([0, 14])



        def animate(i): 
            data.set_data(self.xi[0], np.abs(self.snapshots[i]))
            return [data] # Return what has been modified

        theAnim = animation.FuncAnimation(fig, animate, blit=True, repeat=True) # Note the needed `theAnim` variable. Without it, the garbarge collector would destroy the animation before it is over
        plt.show()    
        return theAnim
    
    def animatepsi2d_save(self, filename = None ):
        if filename != None: 
            path = fr"C:\Users\TQC User\Desktop\BECs\{filename}.mp4"
        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snapshots[0]), extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot)
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
        time_text.set_text('time = 0')
        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        #plt.contour(self.xi[0], self.xi[1], self.Vs) 
        plt.title(f'Animation for L={self.L}')

        def animate(i): 
            data.set_data(np.abs(self.snapshots[i]))
            #time_text.set_text('time = %.1d' % self.time_tracking[i])
            return data, time_text
        anim = animation.FuncAnimation(fig, animate, frames = len(self.snapshots))
        writervideo = animation.FFMpegFileWriter(fps = 5, metadata={'title:':'Wavefunction Animation'})
        num = 0
        if not filename == None: 
            anim.save(path, writer = writervideo)
        else: 
            while os.path.exists(f'breatheranimation_vortexantivL{self.L}_{num}.mp4'): 
                num += 1
            if self.antiV == True: 
                anim.save(f'breatheranimation_vortexantivL{self.L}_{num}.mp4', writer = writervideo) 
            else: 
                anim.save(f'breatheranimation_vortexL{self.L}_{num}.mp4', writer = writervideo)
        return anim 
    
    def animatepsi2d_show(self):
        fig, ax = plt.subplots() 
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, color = 'blue')
        data = plt.imshow(np.abs(self.snapshots[0]), extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot)
        
        time_text.set_text('time = 0')
        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
       # plt.contour(self.xi[0], self.xi[1], self.Vs)
        plt.title(f'Animation for L={self.L}')

        def animate(i): 
            data.set_data(np.abs(self.snapshots[i]))
            time_text.set_text('time = %.1d' % self.time_tracking[i])
            return data
        anim = animation.FuncAnimation(fig, animate, frames = len(self.snapshots))

        plt.show() 
        return anim 
       
        
