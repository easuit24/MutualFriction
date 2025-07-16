import numpy as np

class Point(): 
    def __init__(self, xcoor, ycoor, vortex, trajectory = [], starttime = 0): 
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.vortex = vortex # true if vortex, false if anti-vortex
        self.trajectory = trajectory 
        self.starttime = starttime 
        self.endtime = None 

    def getCoors(self): 
        return (self.xcoor, self.ycoor) 
    
    def getVortexType(self): 
        return self.vortex 
    
    def getTrajectory(self): 
        return np.array(self.trajectory)  
    
    def getStartTime(self): 
        return self.starttime
    
    def calcIndexBuffer(self): 
        '''
        The starting index where the vortex is introduced 
        '''
        return self.starttime/(250*self.dt) 
        
    
    def addCoor(self, x, y): 
        self.xcoor = x
        self.ycoor = y
        self.trajectory.append((x,y))

    def endPoint(self, x, y, end): 
        self.endtime = end
        self.xcoor = x
        self.ycoor = y 





class PointTracker(): 
    def __init__(self, psi_snaps, dx, L, dt, points = [], border_threshold = 4): 
        self.points = points # an array of currently active point objects 
        self.point_history = points # all points that were active during the simulation
        self.unorderedpoints = points 
        self.psi_snaps = psi_snaps
        self.dx = dx
        self.L = L 
        self.dt = dt 
        self.border_threshold = border_threshold
        self.circulation = None
        self.trajectories = []

        self.runTracker()
    
    def getPoints(self):
        return self.points 
    
    def detectVortices(self, psi):
        # extract the inner part of the box
        psi = psi[int(self.L/2/self.dx):int(3*self.L/2/self.dx), int(self.L/2/self.dx):int(3*self.L/2/self.dx)]
    
        Nx, Ny = np.shape(psi)
        S = np.angle(psi)
        vortex_positions = [] 
        anti_vortex_positions = [] 
    
        # initialize arrays
        dS_y_left = np.zeros((len(psi[0]), len(psi[0])))
        dS_y_right = np.zeros((len(psi[0]), len(psi[0])))
        dS_x_top = np.zeros((len(psi[0]), len(psi[0])))
        dS_x_bottom = np.zeros((len(psi[0]), len(psi[0])))
    
        for i in range(self.border_threshold, Nx-self.border_threshold):
            for j in range(self.border_threshold,Ny-self.border_threshold):
                dS_y_left[i,j] = np.mod((S[i, j+1]-S[i,j])+np.pi, 2*np.pi)-np.pi
                dS_y_right[i,j] = np.mod((S[i+1, j+1] - S[i+1, j])+np.pi, 2*np.pi)-np.pi
                dS_x_top[i,j] = np.mod((S[i+1, j+1]-S[i,j+1])+np.pi, 2*np.pi)-np.pi
                dS_x_bottom[i,j] = np.mod((S[i+1,j]-S[i,j])+np.pi, 2*np.pi)-np.pi
                circulation_ij = -dS_y_left[i,j] -dS_x_top[i,j] + dS_y_right[i,j] + dS_x_bottom[i,j] 

                if circulation_ij > 6.2: 
                    vortex_positions.append([(j+0.5)*self.dx, (i+0.5)*self.dx])
                elif circulation_ij < -6.2: 
                    anti_vortex_positions.append([(j+0.5)*self.dx, (i+0.5)*self.dx]) 

        circulation  = -dS_y_left -dS_x_top + dS_y_right + dS_x_bottom  
        

        # plot the circulation to see where the vortices are found!
        self.circulation = circulation 
        return np.array(vortex_positions), np.array(anti_vortex_positions)
    
    def initGrid(self): 
        vp, avp = self.detectVortices(self.psi_snaps[0])
        if len(vp) > 0: # initialize for vortex
            for i in range(len(vp)): 
                self.points.append(Point(vp[i][0], vp[i][1], trajectory = [(vp[i][0], vp[i][1])], vortex = True)) 

        if len(avp) > 0: # initialize for anti-vortex 
            for i in range(len(avp)): 
                self.points.append(Point(avp[i][0], avp[i][1], trajectory = [(avp[i][0], avp[i][1])], vortex = False))

    def getCurrentCoors(self): 
        self.trajectories = [] 
        for i in range(len(self.points)):
            print(self.points[i])

            if np.isnan(self.points[i]): 
                self.trajectories.append(self.points[i].getCoors())
        return self.trajectories 

    def runTracker(self): 
        self.initGrid()


    def newLabelVortices(self): 
        #print(len(self.psi_snaps))
        for i in range(1,len(self.psi_snaps)): 
            
            vortex_positions, anti_vortex_positions = self.detectVortices(self.psi_snaps[i]) # all detected vortices present 
            #all_detected_points = vortex_positions.tolist() + anti_vortex_positions.tolist() 
            all_detected_vortices = vortex_positions.tolist() 
            all_detected_antivortices = anti_vortex_positions.tolist() 
            

            # clause for if the next frame maintains or detects more vortices compared to the previous frame  
            if len(self.points)<= len(vortex_positions) + len(anti_vortex_positions): 
                
                for j in range(len(self.points)): 
    
                    existing_point = self.points[j] 

                    if existing_point.getVortexType() == True and len(vortex_positions)>0: # then it is a vortex 
                        detected_points = vortex_positions

                        euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                        min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                        min_coordinate = vortex_positions[min_index]

                        all_detected_vortices = [x for x in all_detected_vortices if x not in min_coordinate]
                    elif existing_point.getVortexType() == False and len(anti_vortex_positions)>0: 
                        detected_points = anti_vortex_positions
                        euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                        min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                        min_coordinate = anti_vortex_positions[min_index]
                        all_detected_antivortices = [x for x in all_detected_antivortices if x not in min_coordinate]

                # add the "matched" coordinate to the trajectory and update the point in self.points 
                existing_point.addCoor(*min_coordinate[0])                
                self.points[j] = existing_point

            # clause for if a(n) (anti)vortex has disappeared and should be removed from tracking 
            #else: 
                


    

    def getAllVortices(self): 
        '''
        Try to just plot the vortices unordered and see if the vortices are being tracked appropriately 
        '''
        vortex_positions_unordered = np.zeros((len(self.psi_snaps),2)) 
        antivortex_positions_unordered = np.zeros((len(self.psi_snaps), 2)) 



        for i in range(1,len(self.psi_snaps)): 
            
            detected_points = self.detectVortices(self.psi_snaps[i]) # all detected vortices present
            if len(detected_points[0])>0: 
                vortex_positions_unordered[i] = detected_points[0] 

            if len(detected_points[1])>0: 
                
                antivortex_positions_unordered[i] = detected_points[1] 

        return vortex_positions_unordered, antivortex_positions_unordered
    
    def getAllVortices2(self): 
        for i in range(1,len(self.psi_snaps)): 
            
            vortex_positions, anti_vortex_positions = self.detectVortices(self.psi_snaps[i]) # all detected vortices present 

            

            for j in range(len(self.unorderedpoints)): 
                
                # existing_point = self.unorderedpoints[j] 
                # if existing_point.getVortexType() == True: 
                #     detected_points = vortex_positions 
                # else: 
                #     detected_points = anti_vortex_positions 
                


 
                existing_point = self.unorderedpoints[j] 
                if existing_point.getVortexType() == True: # then it is a vortex 
                    detected_points = vortex_positions
                    euclidean_distances = np.abs(existing_point[0] - detected_points[:,0])**2 + np.abs(existing_point[1] - detected_points[:,1])**2
                    min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                    min_coordinate = vortex_positions[min_index]
                else: 
                    detected_points = anti_vortex_positions
                    euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                    min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                    min_coordinate = anti_vortex_positions[min_index]
                existing_point.addCoor(*min_coordinate[0]) 
                self.unorderedpoints[j] = existing_point

    def maintainOrAddVortices(self, vortex_positions, anti_vortex_positions, snapindex): 
        all_detected_antivortices = anti_vortex_positions.tolist() 
        all_detected_vortices = vortex_positions.tolist()
        for j in range(len(self.points)): 
 
            existing_point = self.points[j] 

            if existing_point.getVortexType() == True and len(vortex_positions)>0: # then it is a vortex 
                detected_points = vortex_positions

                euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                min_coordinate = vortex_positions[min_index]

                all_detected_vortices = [x for x in all_detected_vortices if x not in min_coordinate]
            elif existing_point.getVortexType() == False and len(anti_vortex_positions)>0: 
                detected_points = anti_vortex_positions
                euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                min_coordinate = anti_vortex_positions[min_index]

                all_detected_antivortices = [x for x in all_detected_antivortices if x not in min_coordinate]

                
            
            existing_point.addCoor(*min_coordinate[0]) 
            self.points[j] = existing_point 

        #all_detected_points = all_detected_vortices + all_detected_antivortices 
        if len(all_detected_vortices) + len(all_detected_antivortices) > 0: 
            print(snapindex, ": New Vortices Appeared")
            time = 250*snapindex*self.dt 
            for i in range(len(all_detected_vortices)): 
                self.points.append(Point(all_detected_vortices[i][0], all_detected_vortices[i][1], trajectory = [(all_detected_vortices[i][0], all_detected_vortices[i][1])], vortex = True, starttime = time))
            for i in range(len(all_detected_antivortices)): 
                self.points.append(Point(all_detected_antivortices[i][0], all_detected_antivortices[i][1], trajectory = [(all_detected_antivortices[i][0], all_detected_antivortices[i][1])], vortex = False, starttime = time))
        

    def removeVortices(self, vortex_positions, anti_vortex_positions, snapindex): 
        existing_coords = np.zeros((len(self.points),2))
        existing_vortex_coords = [] 
        existing_antivortex_coords = [] 
        for i in range(len(self.points)): 
            if self.points[i].getVortexType() == True: 
                existing_vortex_coords.append(self.points[i].getCoors())
            else: 
                existing_antivortex_coords.append(self.points[i].getCoors()) 
            existing_coords[i] = self.points[i].getCoors() 
        removal_candidates = self.points # must remove from self.points and also set the end time for the point via the corresponding function in Point
        
        for j in range(len(vortex_positions)): # loop over the detected vortices - find the closest vortex match in self.points 
            detected_vortex_coords = vortex_positions[j] 
            print(existing_vortex_coords)
            euclidean_distances = np.sqrt(np.abs(np.array(existing_vortex_coords)[:,0] - detected_vortex_coords[0])**2 + np.abs(np.array(existing_vortex_coords)[:,1] - detected_vortex_coords[1])**2)
            min_index = np.where(euclidean_distances == np.min(euclidean_distances)) 
            min_coordinate = existing_coords[min_index] # the existing point that is a match for the jth detected vortex
            existing_vortex_coords = [x for x in existing_vortex_coords if x not in min_coordinate]

        for j in range(len(anti_vortex_positions)): # loop over the detected vortices - find the closest vortex match in self.points 
            detected_vortex_coords = anti_vortex_positions[j] 
            #print(existing_antivortex_coords[:,0]) 
            #print(detected_vortex_coords[0])
            print(existing_antivortex_coords)
            euclidean_distances = np.sqrt(np.abs(np.array(existing_antivortex_coords)[:,0] - detected_vortex_coords[0])**2 + np.abs(np.array(existing_antivortex_coords)[:,1] - detected_vortex_coords[1])**2)
            min_index = np.where(euclidean_distances == np.min(euclidean_distances)) 
            min_coordinate = existing_coords[min_index] # the existing point that is a match for the jth detected vortex
            existing_antivortex_coords = [x for x in existing_antivortex_coords if x not in min_coordinate]

        # the remaining Point(s) should be removed 
        
        # start with vortices: go through the remaining unclaimed vortices in existing_vortex_coords 
        # compare these with all the coordinates that are currently active in existing_coords - maybe rename this to active coords 
        for i in range(len(existing_vortex_coords)): 
            currtime = 250*snapindex*self.dt
            print("Previous Active Points: ", self.getCurrentCoors())
            #print(np.where(existing_coords[:,0] == existing_vortex_coords[i][0] and existing_coords[:,1] == existing_vortex_coords[i][1]))
            index_to_remove = np.intersect1d(np.where(existing_coords[:,0] == existing_vortex_coords[i][0]), np.where(existing_coords[:,1] == existing_vortex_coords[i][1]))
            #index_to_remove = np.where(existing_coords[:,0] == existing_vortex_coords[i][0] and existing_coords[:,1] == existing_vortex_coords[i][1])
            
            #print(np.where(existing_coords[0] == existing_vortex_coords[i][0] and existing_coords[1] == existing_vortex_coords[i][1]))
            print(index_to_remove[0])
            x,y = self.points[index_to_remove[0]].getCoors()
            self.points[index_to_remove[0]].endPoint(x,y,currtime)
            self.points[index_to_remove[0]] = np.nan
            #del self.points[index_to_remove[0]] # be careful - when you delete index, it shifts the array 
            # compile a list of indices to remove from the array and then just remove them all after the loop 
            print("New Active Points: ", self.getCurrentCoors())
        self.points = [x for x in self.points if str(x) != np.nan]

        
                 

    def labelVortices2(self): 
        for i in range(1,len(self.psi_snaps)): 
            
            vortex_positions, anti_vortex_positions = self.detectVortices(self.psi_snaps[i]) # all detected vortices present 
            all_detected_vortices = vortex_positions.tolist() 
            all_detected_antivortices = anti_vortex_positions.tolist()  

            if len(all_detected_vortices) + len(all_detected_antivortices) >= len(self.points): 
                self.maintainOrAddVortices(vortex_positions, anti_vortex_positions, i)
            else: 
                self.removeVortices(vortex_positions, anti_vortex_positions, i) 


    def labelVortices(self): 
        #print(len(self.psi_snaps))
        for i in range(1,len(self.psi_snaps)): 
            
            vortex_positions, anti_vortex_positions = self.detectVortices(self.psi_snaps[i]) # all detected vortices present 
            #all_detected_points = vortex_positions.tolist() + anti_vortex_positions.tolist() 
            all_detected_vortices = vortex_positions.tolist() 
            all_detected_antivortices = anti_vortex_positions.tolist() 
            for j in range(len(self.points)): 
 
                existing_point = self.points[j] 

                 # clause for if the tracker should add or maintain the number of vortices 
                if existing_point.getVortexType() == True and len(vortex_positions)>0: # then it is a vortex 
                    detected_points = vortex_positions
                    #print("Detected Vortices: ", detected_points)
                    euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                    min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                    min_coordinate = vortex_positions[min_index]
                    #all_detected_vortices.remove(min_coordinate[0]) 
                    all_detected_vortices = [x for x in all_detected_vortices if x not in min_coordinate]
                elif existing_point.getVortexType() == False and len(anti_vortex_positions)>0: 
                    detected_points = anti_vortex_positions
                    euclidean_distances = np.sqrt(np.abs(existing_point.getCoors()[0] - detected_points[:,0])**2 + np.abs(existing_point.getCoors()[1] - detected_points[:,1])**2)
                    min_index = np.where(euclidean_distances == np.min(euclidean_distances))
                    min_coordinate = anti_vortex_positions[min_index]
                    # print(min_coordinate)
                    # print(all_detected_antivortices)
                    all_detected_antivortices = [x for x in all_detected_antivortices if x not in min_coordinate]
                    #all_detected_antivortices.remove(min_coordinate[0])

                # check to see if vortices have disappeared 
                    
                
                existing_point.addCoor(*min_coordinate[0]) 
                #all_detected_points.remove(min_coordinate[0])
                self.points[j] = existing_point 

            #all_detected_points = all_detected_vortices + all_detected_antivortices 
            if len(all_detected_vortices) + len(all_detected_antivortices) > 0: 
                print(i, ": New Vortices Appeared")
                time = 250*i*self.dt 
                for i in range(len(all_detected_vortices)): 
                    self.points.append(Point(all_detected_vortices[i][0], all_detected_vortices[i][1], trajectory = [(all_detected_vortices[i][0], all_detected_vortices[i][1])], vortex = True, starttime = time))
                for i in range(len(all_detected_antivortices)): 
                    self.points.append(Point(all_detected_antivortices[i][0], all_detected_antivortices[i][1], trajectory = [(all_detected_antivortices[i][0], all_detected_antivortices[i][1])], vortex = False, starttime = time))
            
            #########
            # Adding Vortices: More detected vortices than initialized vortices 
            if len(self.points) < len(vortex_positions) + len(anti_vortex_positions): 
                # initialize new points that correspond to the new vortices
                print(i, ': More detected points')  
                print(vortex_positions) 
                print(anti_vortex_positions)


            elif len(self.points) > len(vortex_positions) + len(anti_vortex_positions): 
                # end the points that correspond to the unclaimed points 
                print('Fewer detected points') 
                print(vortex_positions) 
                print(anti_vortex_positions)

        
                 


