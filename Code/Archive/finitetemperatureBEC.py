class GPETimeEv():
 
    def __init__(self, L = 50, dtcoef = 0.1, npoints = 2**9, dim = 1, numVort = 10, spawnType = 'pair', tol = 10e-10, Nfactor = 1000, numImagSteps = 2000, numRealSteps = 10000,  antiV = False, dist = 5, runDyn = True, imp = False, impPsi = None): 
        
        # Simulation parameters 
        self.L = L 
        self.winMult = 2 # multiplier: how much bigger the window is than the condensate
        self.winL = self.winMult*self.L # set the window length for the simulation
        self.npoints = self.winMult*npoints
        self.dx = self.winL/self.npoints 
        self.dk = None
        
        self.tol = tol
        self.imp = imp 
        
        # Physical parameters and grids 
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

        # Vortex parameters 
        self.numVort = numVort # number of vortices to generate 
        if spawnType == 'pair':
            self.numVort = 2

        # Interaction parameters 
        self.g = self.L**2/self.Natoms

        # Time parameters 
        self.dtcoef = dtcoef 
        self.dt = self.dtcoef*(self.dx**2) # coef was originally 0.1
        
        self.Vbox = self.setbox() 

        # Evolution parameters 
        self.numImagSteps = numImagSteps
        self.numRealSteps = numRealSteps
        self.spawnType = spawnType 
        self.antiV = antiV
        self.dist = dist 
        self.runDyn = runDyn

        #vortex tracking parameters
        self.neighborhoodHalfLength = self.npoints//100 
        #self.neighborhoodHalfLength = self.npoints//200 
        #self.neighborhoodHalfLength = self.npoints//10
        self.startvortex_loc = []
        self.vortex_locs = [] 
        self.image_vortex_locs = [] 



        # Arrays 
        self.Ep_arr = []
        self.Ek_arr = [] 
        self.Ei_arr = [] 
        self.time_tracking = [] 

        
        # run the program

        if not self.imp: # if there is no wavefunction import - start from the beginning 
            
            self.initpsi()
            t = time.time() 
            self.simulateimag()
            print("Total Imag Time: ", time.time() - t)
            if self.runDyn == True: 
                self.simulatevortex()
        else:  # start from the wavefunction import 
            self.impPsi = impPsi 
            self.simulatevortex() 

        
        
    def setxgrid(self): 
        '''
        Set up the x grid containing discrete values to evaluate the GPE on. The grid is evaluated over a window around the box potential 
        '''
    
        # set x grid 
        axes = [] 
        for i in range(self.dim): 

            axes.append(np.linspace(-self.winL/2,(self.npoints-1)*self.dx-self.winL/2,self.npoints) )
            
        axes_arr = np.array(axes) 
        self.xi = np.meshgrid(*axes_arr) 
        
    def setpgrid(self): 
        '''
        Set up the momentum (p) grid containing discrete momentum values corresponding to the spatial values defined in setxgrid() 
        '''
        paxes = [] 
        for i in range(self.dim): 
            paxes.append(ifftshift(np.linspace(-np.pi/self.dx, (self.npoints - 1) * (2*np.pi)/self.winL - (np.pi/self.dx), self.npoints)))
        paxes_arr = np.array(paxes) 
        self.ki = np.meshgrid(*paxes_arr) 
        self.dk = 2*np.pi/(self.dx * self.npoints)