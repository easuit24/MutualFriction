import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from scipy import ndimage 
from scipy import constants 

class GPETimeEv():
 
    def __init__(self, L = 50, npoints = 2**9, dim = 1, tol = 10e-10, Nfactor = 1000): 
        # define constants: assume we use Rubidum-87
        self.HBAR = constants.value(u'Planck constant') 
        self.SCAT = 90 * 0.0529e-9
        #self.MASS = 1.44e-25
        self.MASS = 84.911789732 * constants.value(u'atomic mass constant')
        
        self.L = L 
        self.winMult = 1 # multiplier: how much bigger the window is than the condensate
        self.winL = self.winMult*self.L # set the window length for the simulation
        self.npoints = self.winMult*npoints
        self.dx = self.winL/self.npoints 
        self.dk = None
        
        self.tol = tol
        
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
        if self.dim==2:
         self.g = self.L**2/self.Natoms
        else: 
            self.g = self.L/self.Natoms
        #self.g = 0
        #self.g = 1e-5 # remove this later: debugging purposes for energy calculation
        self.g_sho = None
        
        self.dt = 0.1*(self.dx**2) 
        
        self.Vbox = self.setbox() 
        self.Vpert = self.setVpert()
        self.Vs = self.setVsho() 

        self.Ep_arr = []
        self.Ek_arr = [] 
        self.Ei_arr = [] 
        self.virial = [] 

        self.Ep_arr2 = []
        self.Ek_arr2 = [] 
        self.Ei_arr2 = [] 
        self.virial2 = []
       

        self.initpsi()
        
        # run the program 
        self.simulateimag()
        #self.simulatebreather()
        #self.simulatereal()
        
        
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
            paxes.append(np.linspace(-np.pi/self.dx, (self.npoints - 1) * (2*np.pi)/self.winL - (np.pi/self.dx), self.npoints))
        paxes_arr = np.array(paxes) 
        self.ki = np.meshgrid(*paxes_arr) 
        self.dk = 2*np.pi/(self.dx * self.npoints)
        
        
    def setbox(self):
        '''
        Set up the potential energy box to contain the wavefunction assuming a perfect box trap 
        '''
        V = 0
        for i in range(self.dim):
            V += 15 * np.array(np.power(np.e, -2*(self.xi[i] - np.ones_like(self.xi[i])*self.L/2)**2) + np.power(np.e, -2*(self.xi[i] + np.ones_like(self.xi[i])*self.L/2)**2))

        V[V<self.tol] = 0.0 # equivalent of Mathematica's Chop
        return V 
        

    def initpsi(self): 
        psi0 = np.ones_like(self.xi[0])
        
        psi0[np.abs(self.xi[0]) > self.L/2] = 0 # define there to be no atoms outside the box
        if self.dim == 2:
            psi0[np.abs(self.xi[1]) > self.L/2] = 0
        norm = np.sum(np.real(psi0)**2) * self.dx**self.dim

        self.psi_init = np.sqrt(self.Natoms/norm) * psi0
        self.psi = np.sqrt(self.Natoms/norm) * psi0
        
    def setVpert(self, amp = 0.5, width = 0.5): 
        '''
        Defines the potential perturbation function assuming that it takes on the form of a Gaussian
        Parameters: 
        amp - amplitude of the perturbation
        width - width of the perturbation 
    
        Returns: 
        Vp - the perturbation potential function 
        '''
        Vp = 0
        for i in range(self.dim): 
            Vp+= amp * np.power(np.e, -width * self.xi[i]**2)
        Vp[Vp < self.tol] = 0 # Mathematica chop equivalent     
        # plt.figure() 
        # plt.plot(Vp) 
        # plt.show()
        return Vp 
    
    def setVsho(self, w = 0.05): 
        '''
        Sets the harmonic potential energy curve for arbitury dimensions. This 
        is used for the breather case where the optical box is instantaneously 
        switched to a harmonic potential well. 

        Parameters: 
        w - omega; angular frequency of the trap 
        '''
        self.w = w
        Vs = 0
        for i in range(self.dim): 
            Vs += 0.5 * w**2 * self.xi[i]**2 
        #self.g_sho = np.sqrt(8* np.pi) *self.SCAT * np.sqrt(self.MASS * self.w/self.HBAR)
        self.g_sho = 1/(self.w*self.Natoms)

        return Vs

        
    def setk2(self): 
        '''
        Defines the kinetic energy squared based on the values from the momentum grid 
        '''
        self.k2 = 0 
        for i in range(self.dim):
            self.k2 += (self.ki[i]**2)/2

    def lapl(self, arr):
        grad_y, grad_x = np.gradient(arr, self.dx, self.dx) 
        grad_xx = np.gradient(grad_x, self.dx, axis = 1) 
        grad_yy = np.gradient(grad_y, axis = 0)
        return(grad_xx + grad_yy) 

    def calcEnergy(self): 
        '''
        Calculates the potential energy
        '''
        self.r = np.sqrt(self.xi[0]**2 + self.xi[1]**2) 
        self.dr = np.sqrt(2*self.dx**2)

        Ep = self.w**2 / 2 * np.sum(np.cumsum(self.r**2*np.abs(self.dynpsi)**2*self.dr, axis = 1)[:,-1]*self.dr)
        #Ep = self.Natoms * np.sum(np.conj(self.dynpsi) * self.Vs * self.dynpsi)*(self.dx**self.dim)
        self.Ep_arr.append(Ep)

        Ei = self.g_sho * np.sum(np.cumsum(np.abs(self.dynpsi)**4*self.dr, axis = 1)[:,-1]*self.dr)
        self.Ei_arr.append(Ei) 

        #Ek = 0.5 * np.sum(np.cumsum( np.abs(ndimage.laplace(self.dynpsi))**2*self.dr, axis = 1)[:,-1]*self.dr)
        Ek = 0.5 * np.sum(np.cumsum( np.abs(self.lapl(self.dynpsi))**2*self.dr, axis = 1)[:,-1]*self.dr)
        self.Ek_arr.append(Ek)

        self.virial.append(-2*Ek + 2*Ep - 3*Ei) # should be zero

    def calcEnergy2(self): 
        '''
        Calculates the potential energy
        '''
        self.r = np.sqrt(self.xi[0]**2 + self.xi[1]**2) 
        self.dr = np.sqrt(2*self.dx**2)

        Ep = self.MASS * self.w**2 / 2 * np.sum(np.cumsum(self.r**2*np.abs(self.dynpsi)**2*self.dr, axis = 1)[:,-1]*self.dr)
        #Ep = self.Natoms * np.sum(np.conj(self.dynpsi) * self.Vs * self.dynpsi)*(self.dx**self.dim)
        self.Ep_arr2.append(Ep)

        Ei = 0.5 * self.g_sho * np.sum(np.cumsum(np.abs(self.dynpsi)**4*self.dr, axis = 1)[:,-1]*self.dr)
        self.Ei_arr2.append(Ei) 

        #Ek = 0.5 * np.sum(np.cumsum( np.abs(ndimage.laplace(self.dynpsi))**2*self.dr, axis = 1)[:,-1]*self.dr)
        Ek = 0.5 * self.HBAR**2 / self.MASS * np.sum(np.cumsum( np.abs(self.lapl(self.dynpsi))**2*self.dr, axis = 1)[:,-1]*self.dr)
        self.Ek_arr2.append(Ek)

        self.virial2.append(-2*Ek + 2*Ep - 3*Ei) # should be zero   

    def calcEnergy3(self): 
        '''
        Calculates the potential, kinetic, and interaction energy. The kinetic energy is calculated by 
        transforming to momentum space via a Fourier transform. The calculations are unitful. 
        '''
        dF = self.dx**2 / (np.sqrt(2 * np.pi) **2 )
        psik = np.fft.fftshift(np.fft.fftn(self.dynpsi * dF)) # transform to momentum space
        Ep = np.trapz(np.trapz(np.conj(self.dynpsi) * self.Vs * self.dynpsi))*self.dx**2 
        Ek = np.trapz(np.trapz(np.conj(psik) * self.k2 * psik))*self.dk**2 # calculate KE in momentum space
        Ei = 0.5*self.g_sho * np.trapz(np.trapz(np.abs(self.dynpsi)**4))*self.dx**2

        self.Ep_arr.append(Ep)
        self.Ek_arr.append(Ek) 
        self.Ei_arr.append(Ei) 
        


    def simulateimag(self, Nsteps = 5000): 
        '''
        Propagates the wavefunction in imaginary time. Starts from the initial tanh solution guess and uses
        the split step method (Suziki-Trotter expansion) to determine the resulting wavefunction. Reassigns
        psi to the result of this propagation process. 
        '''
            
        kinUim = np.power(np.e, -self.k2*self.dt)
        self.interim_en = [] 
        
        # set up the potential energy part with propagation loop 
        for i in range(Nsteps): 
            #DEBUGGING: made the trap always harmonic: change self.Vs back to self.Vbox
            potUim = np.power(np.e, -(self.Vbox + self.g  * np.abs(self.psi)**2 - 1) *self.dt)
            #potUim = np.power(np.e, -(self.Vs + self.g  * np.real(self.psi) **2 -1) *self.dt) 
            psiFTold = np.fft.fftshift(np.fft.fftn(self.psi)) 
            
            # fix this to make nD later 
            psiFTnew = np.array(psiFTold * kinUim) 
            psiinterim = np.fft.ifftn(np.fft.ifftshift(psiFTnew))
            psinew = potUim * psiinterim 
            norm = np.sum(np.real(psinew)**2) * self.dx**self.dim
            self.psi = np.sqrt(self.Natoms/norm) * psinew
            if (i % 200 == 0 and self.dim == 2) : 
                Ep = np.trapz(np.trapz(self.Vbox * np.abs(self.psi)**2))*self.dx**2

                # kinetic energy
                dF = self.dx**2 / (np.sqrt(2 * np.pi) **2 )
                psik = np.fft.fftshift(np.fft.fftn(self.psi * dF)) # transform to momentum space
                Ek =  np.trapz(np.trapz(self.k2 * np.abs(psik)**2))*self.dk**2
                en = Ep + Ek 
                self.interim_en.append(en)


        
    def simulatereal(self):
        '''
        Propagates the wavefunction in real time for a perturbation in the box. Starts from the result
        of the imaginary time propagation and uses the split step method to evolve the wavefunction
        in real time. It stores snapshots of the wavefunction during the evolution and updates 
        dynpsi. 
        '''
        kinU = np.power(np.e, -(1.0j + 0.005)*(self.k2)*self.dt)
        
        self.snapshots = [self.psi] 
        self.dynpsi = self.psi.copy() # set dynamic psi to evolve in time
        
        for i in range(10000): 

            potU = np.power(np.e, -(1.0j+0.005 ) *((self.Vbox + self.Vpert)+self.g * np.abs(self.dynpsi)**2/2)*self.dt)
            psiFTold = np.fft.fftshift(np.fft.fftn(self.dynpsi))
            psiFTnew = np.array(psiFTold * kinU) 
            psiinterim = np.fft.ifftn(np.fft.ifftshift(psiFTnew))
            psinew = potU * psiinterim 
           # norm = 0 
            
            # for j in range(len(psinew)-1):
                # norm += (np.abs(psinew[j])**2 + np.abs(psinew[j+1])**2)/2 * self.dx ** self.dim
                
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            self.dynpsi = np.sqrt(self.Natoms/norm) * psinew
             

            if (i%250 == 0):
                self.snapshots.append(self.dynpsi)

        self.snapshots = np.array(self.snapshots)  

    def simulatebreather(self):
        '''
        Propagates the wavefunction in real time assuming the system has stablized
        after being confined in an optical box. The potential is instantaneously
        switched to a harmonic potential and the condensate is allowed to evolve.  
        '''
        kinU = np.power(np.e, -(1.0j )*(self.k2)*self.dt)
        
        self.snapshots = [self.psi] 
        self.dynpsi = self.psi.copy() # set dynamic psi to evolve in time
        
        for i in range(10000): # TODO: change this back to 10000 later
            # TODO: potU switch to harmonic potential instead: how would that change the
            # propagator?? 
            potU = np.power(np.e, -(1.0j) *((self.Vs)+self.g_sho * np.abs(self.dynpsi)**2-1)*self.dt)
            #psiFTold = np.fft.fftshift(np.fft.fftn(np.abs(self.dynpsi))) # abs was added here for debugging
            psiFTold = np.fft.fftshift(np.fft.fftn(self.dynpsi))
            psiFTnew = np.array(psiFTold * kinU) 
            psiinterim = np.fft.ifftn(np.fft.ifftshift(psiFTnew))
            psinew = potU * psiinterim 
           # norm = 0 
            
            # for j in range(len(psinew)-1):
                # norm += (np.abs(psinew[j])**2 + np.abs(psinew[j+1])**2)/2 * self.dx ** self.dim
                
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            self.dynpsi = np.sqrt(self.Natoms/norm) * psinew
             

            if (i%250 == 0):
                self.snapshots.append(self.dynpsi)
                self.calcEnergy3() # calculates potential energy and add to array

        self.snapshots = np.array(self.snapshots)

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
    
    def animatepsi2d_save(self):
        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snapshots[0]), extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot)
        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.contour(self.xi[0], self.xi[1], self.Vs) 
        plt.title(f'Animation for w={self.w}, L={self.L}')

        def animate(i): 
            data.set_data(np.abs(self.snapshots[i]))
            return [data]
        anim = animation.FuncAnimation(fig, animate, frames = len(self.snapshots))
        writervideo = animation.FFMpegFileWriter()
        anim.save(f'breatheranimation_w{self.w}L{self.L}.mp4', writer = writervideo)
        #plt.show() 
        return anim 
    
    def animatepsi2d_show(self):
        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snapshots[0]), extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot)
        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.contour(self.xi[0], self.xi[1], self.Vs)
        plt.title(f'Animation for w={self.w}, L={self.L}')

        def animate(i): 
            data.set_data(np.abs(self.snapshots[i]))
            return [data]
        anim = animation.FuncAnimation(fig, animate)

        plt.show() 
        return anim 
       
        
