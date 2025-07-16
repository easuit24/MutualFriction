import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera 
from scipy import constants 
from scipy.fft import fft2, ifft2, fftshift, ifftshift 
from scipy.ndimage import minimum_filter
from skimage.feature import peak_local_max
import time 
import os 

class GPETimeEv():
 
    def __init__(self, L = 20, dtcoef = 0.1, npoints = 2**9, dim = 1, winMult = 2, numVort = 10, boxthickness = 2, spawnType = 'No', tol = 10e-10, Nfactor = 1000, numImagSteps = 2000, numRealSteps = 10000,  antiV = False, dist = 2, runDyn = True, imp = False, impPsi = None): 
        
        # Simulation parameters 
        self.L = L 
        self.winMult = winMult # multiplier: how much bigger the window is than the condensate
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
        
        self.thickness = boxthickness 
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
        self.kmax = np.pi/self.dx
        
        
    def setbox2(self):
        '''
        Set up the potential energy box to contain the wavefunction assuming a box potential  
        '''
        V = 0
        self.height = 5
        self.height = self.kmax**2/4
        #self.height = self.kmax**2/2 # adjust this height coefficient as needed 
        self.steepness = 2
        #self.height =5
        for i in range(self.dim):
            V += self.height * np.array(np.power(np.e, -self.steepness*(self.xi[i] - np.ones_like(self.xi[i])*self.L/2)**2) + np.power(np.e, -self.steepness*(self.xi[i] + np.ones_like(self.xi[i])*self.L/2)**2))
            
        V[V<self.tol] = 0.0 # set values close to 0 to be 0
        # V[np.abs(self.xi[0])>=self.L//2] = self.height 
        # V[np.abs(self.xi[1])>=self.L//2] = self.height
    
        return V 
    
    def setbox(self):
        #shape=(100, 100), L=40, height=1, thickness=1
        shape = np.shape(self.xi[0])
        L = self.L/self.dx 
        # *2.5 seemed to work pretty ok with some flow out of the box
        height = self.kmax**2/2*3
        thickness = self.thickness 

        x = np.linspace(-shape[0]//2, shape[0]//2, shape[0])
        y = np.linspace(-shape[1]//2, shape[1]//2, shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        half_L = L / 2

        def edge_strip(coord, center, thickness):
            return np.exp(-((coord - center) ** 2) / (2 * thickness**2))

        # Create masked edge strips
        top_edge    = edge_strip(Y,  half_L, thickness) * (np.abs(X) <= half_L)
        bottom_edge = edge_strip(Y, -half_L, thickness) * (np.abs(X) <= half_L)
        left_edge   = edge_strip(X, -half_L, thickness) * (np.abs(Y) <= half_L)
        right_edge  = edge_strip(X,  half_L, thickness) * (np.abs(Y) <= half_L)


        # Combine edges using maximum to prevent corner stacking
        V = height * np.maximum.reduce([top_edge, bottom_edge, left_edge, right_edge])


        return V
        

    def initpsi(self): 
        '''
        Initializes the wavefunction to a constant within the trap and zero outside of the trap. Spawns the specified vortices according to the inputs to the
        class 

        '''
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
        else: 
            print("No vortices to be spawned")
            
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
        self.startvortex_loc = [(0,0), (x_loc, y_loc)]

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
        for v in locs: 
            x_loc, y_loc = v 
            print("Spawning Vortex at: ", x_loc, ",", y_loc)

            r_grid += np.sqrt((x_loc-self.xi[0])**2 + (y_loc-self.xi[1])**2) # set the vortex to appear at v

            if not antiV: 
                theta_grid += np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0])) # set the phase at the vortex at v assuming vortex 
            else: 
                theta_grid -= np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0])) # set the phase at the vortex at v assuming antivortex 

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
        x_loc, y_loc = np.random.uniform(-self.L/2,self.L/2), np.random.uniform(-self.L/2,self.L/2)
        r_grid = np.sqrt((x_loc-self.xi[0])**2 + (y_loc-self.xi[1])**2)
        if antiV == False: # for vortices 
            for v in range(int(prop*self.numVort)): 
                theta_grid += np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0]))

        else: # for antivortices 
            for v in range(int(prop*self.numVort)): 
                theta_grid -= np.arctan2((y_loc-self.xi[1]),(x_loc-self.xi[0]))

        psiVort = psi * np.tanh(r_grid)*np.exp(theta_grid * 1j)
        return psiVort
    
    def initNeighborhood(self): 
        '''
        Initialize the neighborhood by determining the center of each vortex and the corresponding region of interest "searchArea" for each vortex 
        '''
        axis = self.xi[0][0] 

        # first vortex
        xstart_ind = self.winL//2/self.dx - self.neighborhoodHalfLength
        ystart_ind = self.winL//2/self.dx - self.neighborhoodHalfLength
        xend_ind = xstart_ind + 2*self.neighborhoodHalfLength 
        yend_ind = ystart_ind + 2*self.neighborhoodHalfLength
        self.inds = np.array([xstart_ind, ystart_ind, xend_ind, yend_ind])
        self.xneighborhoodAxis = axis[int(xstart_ind):int(xend_ind)+1]
        self.yneighborhoodAxis = axis[int(ystart_ind):int(yend_ind)+1]
        
        # initialize ROI 
        self.searchArea = np.abs(self.dynpsi[int(ystart_ind):int(yend_ind),int(xstart_ind):int(xend_ind)]) 

        # second vortex 
        xstart_ind2 = xstart_ind + self.dist/self.dx
        ystart_ind2 = ystart_ind 
        xend_ind2 = xend_ind + self.dist/self.dx
        yend_ind2 = yend_ind # since the vortex is along the x axis - generalize this later 
        self.inds2 = np.array([xstart_ind2, ystart_ind2, xend_ind2, yend_ind2])        
        self.xneighborhoodAxis2 = axis[int(xstart_ind2):int(xend_ind2)+1]
        self.yneighborhoodAxis2 = axis[int(ystart_ind2):int(yend_ind2)+1]

        # initalize ROI 
        self.searchArea2 = np.abs(self.dynpsi[int(ystart_ind2):int(yend_ind2),int(xstart_ind2):int(xend_ind2)])

        # find minima to determine the center of the window 

        minima = peak_local_max(-self.searchArea, threshold_abs=-5, exclude_border = False)
        self.minima = minima
        winCenter = (len(self.searchArea)//2, len(self.searchArea)//2)
        minima = self.multipleVortices(minima, winCenter) 

        self.vort_coords = (minima[1], minima[0])
        minima2 = peak_local_max(-self.searchArea2, threshold_abs=-5, exclude_border = False)
        winCenter2 = (len(self.searchArea2)//2, len(self.searchArea2)//2)
        minima2 = self.multipleVortices(minima2, winCenter2) 

        self.vort_coords2 = (minima2[1], minima[0])

        self.centers = np.array([[minima[1], minima[0]], [minima2[1], minima2[0]]])



    def trackVortices(self): 
        '''
        Tracks vortices by finding the minimum wavefunction in the given ROI and adjusts the ROI if necessary. Also accounts for multiple vortices in a given
        window and determines which vortex is the one that is tracked based on the minimum distance from the previous vortex coordinate. Assumes a pair of 
        vortices 
        '''

        # unpack index arrays for clarity
        xstart_ind = self.inds[0]
        ystart_ind = self.inds[1]
        xend_ind = self.inds[2]
        yend_ind = self.inds[3] 

        xstart_ind2 = self.inds2[0]
        ystart_ind2 = self.inds2[1]
        xend_ind2 = self.inds2[2]
        yend_ind2 = self.inds2[3] 

        # define the ROIs 
        self.searchArea = np.abs(self.dynpsi[int(ystart_ind):int(yend_ind),int(xstart_ind):int(xend_ind)]) 
        self.searchArea2 = np.abs(self.dynpsi[int(ystart_ind2):int(yend_ind2),int(xstart_ind2):int(xend_ind2)])

        minima = peak_local_max(-self.searchArea, threshold_abs=-5, exclude_border = False)
        indices = self.multipleVortices(minima, self.vort_coords)

        self.vort_coords = (indices[1], indices[0])

        # ROI for vortex #2 
        minima2 = peak_local_max(-self.searchArea2, threshold_abs=-5, exclude_border = False)
        indices2 = self.multipleVortices(minima2, self.vort_coords2)

        self.vort_coords2 = (indices2[1], indices2[0])

        self.vortex_locs.append([self.vort_coords, self.vort_coords2]) # store locations in an array (vortex 1, vortex 2)

        # update the search region for vortex #1: put this first 
        if self.vort_coords[0] == 0 or self.vort_coords[1] == 0 or self.vort_coords[0] == len(self.searchArea)-1 or self.vort_coords[1] == len(self.searchArea[0])-1: 
            x_diff = self.centers[0][0] - self.vort_coords[0] # define the center to be around the initial vortex center   
            y_diff = self.centers[0][1] - self.vort_coords[1]
            xstart_ind = xstart_ind - x_diff 
            xend_ind = xstart_ind + 2*self.neighborhoodHalfLength
            ystart_ind = ystart_ind - y_diff 
            yend_ind = ystart_ind + 2*self.neighborhoodHalfLength
            self.inds = np.array([xstart_ind, ystart_ind, xend_ind, yend_ind]) # redefine the indices 

            self.xneighborhoodAxis = self.xi[0][0][int(xstart_ind):int(xend_ind)+1]
            self.yneighborhoodAxis = self.xi[0][0][int(ystart_ind):int(yend_ind)+1]
            self.searchArea = np.abs(self.dynpsi[int(ystart_ind):int(yend_ind),int(xstart_ind):int(xend_ind)]) # update the search area 

            minima = peak_local_max(-self.searchArea, threshold_abs=-5, exclude_border = False)
            indices = self.multipleVortices(minima, self.vort_coords)

            self.centers[0] = [indices[1], indices[0]]
            self.vort_coords = (indices[1], indices[0])
            
            self.vortex_locs = self.vortex_locs[:-1]
            self.vortex_locs.append([self.vort_coords, self.vort_coords2])

        # for the second vortex 
        if self.vort_coords2[0] == 0 or self.vort_coords2[1] == 0 or self.vort_coords2[0] == len(self.searchArea2)-1 or self.vort_coords2[1] == len(self.searchArea2[0])-1: 
            x_diff = self.centers[1][0] - self.vort_coords2[0] # define the center to be around the initial vortex center  
            y_diff = self.centers[1][1] - self.vort_coords2[1]

            xstart_ind2 = xstart_ind2 - x_diff 
            xend_ind2 = xstart_ind2 + 2*self.neighborhoodHalfLength
            ystart_ind2 = ystart_ind2 - y_diff 
            yend_ind2 = ystart_ind2 + 2*self.neighborhoodHalfLength

            self.inds2 = np.array([xstart_ind2, ystart_ind2, xend_ind2, yend_ind2]) # redefine the indices
            self.xneighborhoodAxis2 = self.xi[0][0][int(xstart_ind2):int(xend_ind2)+1]
            self.yneighborhoodAxis2 = self.xi[0][0][int(ystart_ind2):int(yend_ind2)+1]
            self.searchArea2 = np.abs(self.dynpsi[int(ystart_ind2):int(yend_ind2),int(xstart_ind2):int(xend_ind2)]) # update the search area 
            
            minima2 = peak_local_max(-self.searchArea2, threshold_abs=-5, exclude_border = False)
            indices2 = self.multipleVortices(minima2, self.vort_coords2)

            self.centers[1] = [indices2[1], indices[0]]

            self.vort_coords2 = (indices2[1], indices2[0])
            self.vortex_locs = self.vortex_locs[:-1]
            self.vortex_locs.append([self.vort_coords, self.vort_coords2])

        # append the final vortex coordinates to the array after the checks are complete 
        self.image_vortex_locs.append([(self.vortex_locs[-1][0][0] + xstart_ind, self.vortex_locs[-1][0][1] + ystart_ind),(self.vortex_locs[-1][1][0]+xstart_ind2,self.vortex_locs[-1][1][1]+ystart_ind2)])

 
    def multipleVortices(self, minima, vorts): 
        '''
        Function to determine which minima corresponds to the vortex in the specified ROI if there are multiple minima in the ROI 

        Parameters: 
        minima - the minima found in the ROI (array) 
        vorts - the coordinates of the vortex tracked in that ROI (tuple) 

        Returns: 
        indices - the indices of the minimum that corresponds to the vortex 
        '''
        # make it relative to the centers 
        if len(minima) > 1: 

            self.minima = minima

            min_dist = np.sqrt((minima[0][1] - vorts[0])**2+(minima[0][0]-vorts[1])**2)
            min_coor = minima[0]  
            # find minimum distance and corresponding coordinate 
            for m in minima: 
                d = np.sqrt((m[1] - vorts[0])**2+(m[0]-vorts[1])**2)
                if d < min_dist:
                    min_dist = d 
                    min_coor = m 

            indices = min_coor 
        elif len(minima) == 1:
            indices = minima[0]
            
        else: 
            print("No More Minima...")
            indices = [(1,1)]
        return indices 
    
    def mapVortexDist(self): 
        '''
        Tracks the distances between the pair of vortices 

        Returns: 
        dist_arr - the distances at each time step between the vortex pair 
        '''

        # Calculate distance 
        dist_arr = np.zeros(len(self.image_vortex_locs))
        i = 0 
        for coor in self.image_vortex_locs: 
            d = np.sqrt((coor[0][0] - coor[1][0])**2 + (coor[0][1] - coor[1][1])**2)
            dist_arr[i] = d 
            i += 1
        dist_arr = dist_arr * self.dx 
        
        return dist_arr
    
    def mapVortexAngle(self): 
        '''
        Tracks the angle between the pair of vortices 

        Returns: 
        angle_arr - the angle in radians between the vortex pair at each time step
        '''

        angle_arr = np.zeros(len(self.image_vortex_locs))
        i = 0 
        for coor in self.image_vortex_locs:
            y = coor[0][1] - coor[1][1] # y separation
            x = coor[1][0] - coor[0][0] # x separation

            # calculat the angle 
            ang = np.arctan(y/x)
            angle_arr[i] = ang
            i+=1

        return angle_arr 

    def calcEnergy(self): 
        '''
        Calculates the potential, kinetic, and interaction energy. The kinetic energy is calculated by 
        transforming to momentum space via a Fourier transform. The energies are stored in three arrays corresponding to the three energy types.  
        '''
        dF = self.dx**2 / (np.sqrt(2 * np.pi) **2 )
        psik = fftshift(fft2(self.dynpsi * dF))
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
        kinUim = np.exp(-self.k2*self.dt) 
        
        # set up the potential energy part with propagation loop 

        for i in range(self.numImagSteps): 

            t0 = time.time() 
            potential = self.Vbox + self.g  * np.abs(self.psi)**2 - 1
            potUim = np.exp(-(potential) *self.dt)
            t1 = time.time() 
            #print(t1-t0) 
            psiFTold = fft2(self.psi) 
            t2 = time.time() 
            #print(t2-t1)
            psiFTnew = psiFTold * kinUim
            t3 = time.time() 
            #print(t3-t2)
            psiinterim = ifft2(psiFTnew)
            t4 = time.time() 
            #print(t4-t3)
            psinew = potUim * psiinterim 
            t5 = time.time() 
            #print(t5-t4)
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            t6 = time.time() 
            #print(t6-t5)
            self.psi = np.sqrt(self.Natoms/norm) * psinew
            t7 = time.time() 
            #print(t7-t6)
            #print("Total: ", t7-t0)



 

    def simulatevortex(self):
        '''
        Propagates the wavefunction in real time assuming the system has stablized
        after being confined in an optical box. The potential is instantaneously
        switched to a harmonic potential and the condensate is allowed to evolve.  
        '''
        kinU = np.exp( -(1.0j )*(self.k2)*self.dt)
        
        self.snapshots = [self.psi] 
        self.time_tracking = [0]
        if not self.imp: 
            self.dynpsi = self.psi.copy()
        else: 
            self.dynpsi = self.impPsi # set the dynamic wavefunction to be the imported one 
        if self.spawnType != 'No':
            self.initNeighborhood() 
        for i in range(self.numRealSteps): 

            potU = np.exp(-(1.0j) *((self.Vbox)+self.g * np.abs(self.dynpsi)**2-1)*self.dt)

            psiFTold = fft2(self.dynpsi)
            psiFTnew = psiFTold * kinU 
            psiinterim = ifft2(psiFTnew)
            psinew = potU * psiinterim 
                
            norm = np.sum(np.abs(psinew)**2) * self.dx**self.dim
            self.dynpsi = np.sqrt(self.Natoms/norm) * psinew
             

            if (i%1000 == 0):
            
                self.snapshots.append(self.dynpsi)
                if self.spawnType != 'No': 
                    self.trackVortices()
                self.calcEnergy() # calculates potential energy and add to array
                self.time_tracking.append(self.dt * i)

        self.snapshots = np.array(self.snapshots)
        if self.spawnType != 'No': 
            self.vortex_locs = np.array(self.vortex_locs)
            self.image_vortex_locs = np.array(self.image_vortex_locs)
            self.dist_arr = self.mapVortexDist() 
            self.angle_arr = self.mapVortexAngle() 
        self.time_tracking = np.array(self.time_tracking) 

        # find the distance and angle between the vortex pair 
        


    
    def animatepsi2d(self, filename):
        if filename != None: 
            path = fr"C:\Users\TQC User\Desktop\BECs\{filename}.mp4"

        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.snapshots[0]), extent = [-self.winL/2, self.winL/2, -self.winL/2, self.winL/2],cmap = plt.cm.hot)
        v1 = plt.scatter(self.xi[0][0][int(self.image_vortex_locs[0][0][0])], self.xi[0][0][int(self.image_vortex_locs[0][0][1])], color = 'blue', marker = '<', s = 20, alpha = 0.3)
        v2 = plt.scatter(self.xi[0][0][int(self.image_vortex_locs[0][1][0])], self.xi[0][0][int(self.image_vortex_locs[0][1][1])], color = 'blue', marker = '>', s = 20, alpha = 0.3)
        
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
        time_text.set_text('time = 0')
        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        # plt.contour(self.xi[0], self.xi[1], self.Vs)
        plt.title(f'Animation for L={self.L}')

        def animate(i): 
            data.set_data(np.abs(self.snapshots[i]))
            v1.set_offsets([self.xi[0][0][int(self.image_vortex_locs[i][0][0])], self.xi[0][0][int(len(self.xi[0][0])-self.image_vortex_locs[i][0][1])]])

            ##self.image_vortex_locs[:,0][i]
            ##v2.set_offsets(self.image_vortex_locs[:,-1][i])
            v2.set_offsets([self.xi[0][0][int(self.image_vortex_locs[i][1][0])], self.xi[0][0][int(len(self.xi[0][0])-self.image_vortex_locs[i][1][1])]])
            time_text.set_text('time = %.1d' % self.time_tracking[i])
            return data, v1, v2, time_text
        anim = animation.FuncAnimation(fig, animate, frames = len(self.image_vortex_locs), blit = True)

        plt.show() 
        
        anim.save(path)

        return anim
       
        
