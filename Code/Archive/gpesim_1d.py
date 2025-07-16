import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import itertools

class GPETimeEv():
 
    def __init__(self, L = 50, npoints = 2**9, dim = 1, tol = 10e-10, omega = 2 * np.pi *19.3): 

        self.L = L 
        self.npoints = npoints
        self.dx = self.L/self.npoints 
        
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
        self.Natoms = 10*self.npoints # set the number of atoms
        self.g = self.L/self.Natoms
        self.omega = omega
        
        self.dt = 0.1*(self.dx**2) 
        
        self.Vbox = self.setbox() 
        self.Vpert = self.setVpert()
        self.Vs = self.setVsho() 

        self.Ep_arr = []

        self.initpsi()
        
        # run the program 
        self.simulateimag()
        #self.simulatebreather()
        self.simulatereal()
        
        
    def setxgrid(self): 
        '''
        Set up the x grid containing discrete values to evaluate the GPE on 
        '''
    
        # set x grid 
        axes = [] 
        for i in range(self.dim): 
            axes.append(np.linspace(-self.L/2,(self.npoints-1)*self.dx-self.L/2,self.npoints) )
            
        axes_arr = np.array(axes) 
        self.xi = np.meshgrid(*axes_arr) 
        
    def setpgrid(self): 
        '''
        Set up the momentum (p) grid containing discrete momentum values corresponding to the spatial values defined above
        '''
        paxes = [] 
        for i in range(self.dim): 
            paxes.append(np.linspace(-np.pi/self.dx, (self.npoints - 1) * (2*np.pi)/self.L - (np.pi/self.dx), self.npoints))
        paxes_arr = np.array(paxes) 
        self.ki = np.meshgrid(*paxes_arr) 
        
        
    def setbox(self):
        '''
        Set up the potential energy box to contain the wavefunction assuming a perfect box trap 
        '''
        V = 0
        for i in range(self.dim):
            V += 5 * np.array(np.power(np.e, -2*(self.xi[i] - np.ones_like(self.xi[i])*self.L/2)**2) + np.power(np.e, -2*(self.xi[i] + np.ones_like(self.xi[i])*self.L/2)**2))

        V[V<self.tol] = 0.0 # equivalent of Mathematica's Chop
        return V 
        
    def initpsi3(self): 
        '''
        Initializes the wavefunction, assumes that a good guess for solving the GPE takes on the form of a tanh function 
        '''
        psi0 = np.zeros_like(self.xi) 

        for i in range(self.dim): 
            psi0 += np.sqrt(self.Natoms/self.L) * (-np.tanh(0.25 * (self.xi[i]-self.L/2)) * np.tanh(0.25*(self.xi[i]+self.L/2)))
        norm = np.sum(np.real(psi0)**2) * self.dx**self.dim 

        self.psi_init = np.sqrt(self.Natoms/norm) * psi0
        self.psi = np.sqrt(self.Natoms/norm) * psi0  

    def initpsi2(self): 
        r2 = 0 
        for i in range(self.dim):
            r2 += self.xi[i]**2
        r = np.sqrt(r2) 
        psi0 = np.sqrt(self.Natoms/self.L) * (-np.tanh(0.25 * (r-self.L/2)) * np.tanh(0.25*(r+self.L/2)))
        norm = np.sum(np.abs(psi0)**2) * self.dx**self.dim 
        self.psi_init = np.sqrt(self.Natoms/norm) * psi0
        self.psi = np.sqrt(self.Natoms/norm) * psi0

    def initpsi(self): 
        psi0 = np.ones_like(self.xi[0])
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
        Vs = 0
        for i in range(self.dim): 
            Vs += 0.5 * w**2 * self.xi[i]**2 

        return Vs

        
    def setk2(self): 
        '''
        Defines the kinetic energy squared based on the values from the momentum grid 
        '''
        self.k2 = 0 
        for i in range(self.dim):
            self.k2 += (self.ki[i]**2)/2

    def calcEnergy(self): 
        '''
        Calculates the potential energy
        '''
        self.r = np.sqrt(self.xi[0]**2 + self.xi[1]**2) 
        self.dr = np.sqrt(2*self.dx**2)
        Ep = self.omega**2 / 2 * np.cumsum(np.cumsum(self.r**2*np.abs(self.dynpsi)**2*self.dr)*self.dr)[-1]
        #Ep = self.Natoms * np.sum(np.conj(self.dynpsi) * self.Vs * self.dynpsi)*(self.dx**self.dim)
        self.Ep_arr.append(Ep)


    def simulateimag(self, Nsteps = 10000): 
        '''
        Propagates the wavefunction in imaginary time. Starts from the initial tanh solution guess and uses
        the split step method (Suziki-Trotter expansion) to determine the resulting wavefunction. Reassigns
        psi to the result of this propagation process. 
        '''
            
        kinUim = np.power(np.e, -self.k2*self.dt)
        
        # set up the potential energy part with propagation loop 
        for i in range(Nsteps): 
            #DEBUGGING: made the trap always harmonic
            potUim = np.power(np.e, -(self.Vbox + self.g  * np.real(self.psi) **2 -1) *self.dt)
            #potUim = np.power(np.e, -(self.Vs + self.g  * np.real(self.psi) **2 -1) *self.dt) 
            psiFTold = np.fft.fftshift(np.fft.fftn(self.psi)) 
            
            # fix this to make nD later 
            psiFTnew = np.array(psiFTold * kinUim) 
            psiinterim = np.fft.ifftn(np.fft.ifftshift(psiFTnew))
            psinew = potUim * psiinterim 
            norm = np.sum(np.real(psinew)**2) * self.dx**self.dim
            self.psi = np.sqrt(self.Natoms/norm) * psinew

        
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

            potU = np.power(np.e, -(1.0j + 0.005) *((self.Vbox + self.Vpert)+self.g * np.abs(self.dynpsi)**2/2-1)*self.dt)
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
        kinU = np.power(np.e, -(1.0j + 0.005)*(self.k2)*self.dt)
        
        self.snapshots = [self.psi] 
        self.dynpsi = self.psi.copy() # set dynamic psi to evolve in time
        
        for i in range(20000): # TODO: change this back to 10000 later
            # TODO: potU switch to harmonic potential instead: how would that change the
            # propagator?? 
            potU = np.power(np.e, -(1.0j+0.005) *((self.Vs)+self.g * np.abs(self.dynpsi)**2/2-1)*self.dt)
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
                self.calcEnergy() # calculates potential energy and add to array

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
    
    def animatepsi2d(self):
        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snapshots[0]), cmap = plt.cm.hot)

        def animate(i): 
            data.set_data(np.abs(self.snapshots[i]))
            return [data]
        anim = animation.FuncAnimation(fig, animate)
        plt.show() 
        return anim 
       
        
