import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation 
from scipy.spatial.distance import cdist

# import classical field module 
from classicalfield_orig import FiniteTempGPE as gpe 

class VortexTracker(): 

    def __init__(self, psi_snaps, L, dx, border_threshold = 4, plot = True, animFileName = 'default'): 
        self.psi_snaps = psi_snaps
        self.L = L 
        self.dx = dx 
        self.border_threshold = border_threshold 
        self.animFileName = animFileName
        self.circ_frames = np.zeros((len(self.psi_snaps), len(self.psi_snaps[0])//2, len(self.psi_snaps[0])//2))
        self.plot = plot 
        self.main()
        self.animatepsi2_vortices(animFileName)
        self.generalAnimation(animFileName, self.circ_frames)

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
                        #anti_vortex_positions.append([(j)*self.dx, (i)*self.dx])

            circulation  = -dS_y_left -dS_x_top + dS_y_right + dS_x_bottom  
            

            # plot the circulation to see where the vortices are found!
            self.circulation = circulation 
            return np.array(vortex_positions), np.array(anti_vortex_positions), circulation
    
    def track_vortices_across_frames2(self, frames, max_dist=5):
        """
        frames: list of lists of (x, y) tuples
        Returns: dict {vortex_id: [(t, x, y), ...]}
        """
        next_id = 0
        tracks = {}  # {id: [(t, x, y)]}
        prev_positions = {}  # {id: (x, y)}

        # Initialize with frame 0
        for coord in frames[0]:
            tracks[next_id] = [(0, *coord)]
            prev_positions[next_id] = coord
            next_id += 1

        for t in range(1, len(frames)):
            current_coords = np.array(frames[t])
            if len(current_coords) == 0: # case for if no vortices are found in a frame 
                continue

            # Match to previous positions 
            prev_ids = list(prev_positions.keys())
            prev_coords = np.array([prev_positions[i] for i in prev_ids]) # extract the coordinates for all active vortices 

            distance_matrix = cdist(prev_coords, current_coords) # find distances 

            self.distance_matrix = distance_matrix
            #matched_ids = set()
            used_indices = set()
            new_prev_positions = {}

            for i, row in enumerate(distance_matrix):
                j = np.argmin(row) # get the index of the minimum distance in the row... but now it might assign things to that vortex that are actuallly closer to a different vortex...
                if row[j] < max_dist and j not in used_indices:
                    vortex_id = prev_ids[i]
                    x, y = current_coords[j]
                    tracks[vortex_id].append((t, x, y))
                    new_prev_positions[vortex_id] = (x, y)
                    #matched_ids.add(vortex_id)
                    used_indices.add(j)

            # Add new vortices (unmatched)
            for j, coord in enumerate(current_coords):
                if j not in used_indices:
                    tracks[next_id] = [(t, *coord)]
                    new_prev_positions[next_id] = coord
                    next_id += 1

            prev_positions = new_prev_positions

        return tracks
    

    def track_vortices_across_frames(self, frames, initial_locs = [(0,0), (3,0)], max_dist=10):
        """
        frames: list of lists of (x, y) tuples
        Returns: dict {vortex_id: [(t, x, y), ...]}
        """
        next_id = 0
        tracks = {}  # {id: [(t, x, y)]}
        prev_positions = {}  # {id: (x, y)}

        # Initialize with initial states set in the vortex generation 
        tracks[0] = [(0, initial_locs[0][0]+self.L/2, initial_locs[0][1]+self.L/2)]
        tracks[1] = [(0, initial_locs[1][0]+self.L/2, initial_locs[1][1]+self.L/2)]

        prev_positions[0] = (initial_locs[0][0]+self.L/2, initial_locs[0][1]+self.L/2)
        prev_positions[1] = (initial_locs[1][0]+self.L/2, initial_locs[1][1]+self.L/2)

        # Iterate over all the detected coordinates in each of the frames
        for t in range(0, len(frames)):
            current_coords = np.array(frames[t])
            if len(current_coords) < 2: # case for if no vortices are found in a frame 
                continue

            # Match to previous positions 
            prev_ids = list(prev_positions.keys())
            prev_coords = np.array([prev_positions[i] for i in prev_ids]) # extract the coordinates for all active vortices 

            if len(prev_coords) < 2: 
                #print(prev_coords)
                continue 

            distance_matrix = cdist(prev_coords, current_coords) # find distances 


            self.distance_matrix = distance_matrix

            used_indices = set()
            new_prev_positions = {}

            for i, row in enumerate(distance_matrix):
                j = np.argmin(row) # get the index of the minimum distance in the row... but now it might assign things to that vortex that are actuallly closer to a different vortex...
                if row[j] < max_dist and j not in used_indices:
                    vortex_id = prev_ids[i]
                    x, y = current_coords[j]

                    tracks[vortex_id].append((t+1, x, y))
                    new_prev_positions[vortex_id] = (x, y)
                    used_indices.add(j)

                # for if the coordinate goes to the other vortex in the pair: go to the next nearest coordinate 
                elif row[j] < max_dist and j in used_indices: 
                    current_coords_subset = np.delete(current_coords.copy(), j, axis = 0) 
                    row_subset = row.copy() 
                    new_min_ind = np.argmin(np.delete(row_subset, j)) 
                    vortex_id = prev_ids[i] 
                    print("Current Coords: ", current_coords) 
                    print("Current Coords Subset: ", current_coords_subset)
                    x,y = current_coords_subset[new_min_ind] # must also modify the distance matrix because now things are shifted...
                    tracks[vortex_id].append((t+1,x,y)) 
                    new_prev_positions[vortex_id] = (x,y) 
                    used_indices.add(new_min_ind)
                    


            # if len(prev_positions) < 2: 
            #     print("Less than 2 previous vortices...")
            #     print(prev_positions) 
            #     print(current_coords)
            prev_positions = new_prev_positions

        return tracks

    def compute_distance_between_tracks(self, track1, track2):
        # to do: figure out how to map arrays that are not the same length but overlap in time
        distances = []
        angles = [] 
        for (t1, x1, y1), (t2, x2, y2) in zip(track1, track2):
            if t1 == t2:
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append((t1, dist))

                ang = np.arctan2(np.abs(y2-y1),np.abs(x2-x1))
                angles.append((t1, ang))
        return distances, angles

    def plot_distance_over_time(self, distances):
        if not distances:
            print("No overlapping time points to compute distance.")
            return
        times, dists = zip(*distances)
        plt.plot(times, dists, marker='o')
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.title("Distance Between Vortices Over Time")
        plt.grid(True)
        plt.show()

    def plot_angle_over_time(self, angles):
        if not angles:
            print("No overlapping time points to compute angles.")
            return
        times, angs = zip(*angles)
        plt.plot(times, angs, marker='o')
        plt.xlabel("Frame")
        plt.ylabel("Angle [rad]")
        plt.title("Angle Between Vortices Over Time")
        plt.grid(True)
        plt.show()


    def animatepsi2_vortices(self, filename): 

        vortex1_traj = self.vortex1
        vortex2_traj = self.vortex2 


        ## TODO: Fix this to add time via a dt input somewhere 
        #time_tracking = np.arange(0, len(self.psi_snaps))*250*
        if filename != None: 
                path = fr"C:\Users\TQC User\Desktop\BECs2\{filename}.mp4"
        fig, ax = plt.subplots() 
        data = plt.imshow(np.abs(self.psi_snaps[0])**2, extent = [-self.L, self.L, -self.L, self.L], cmap = plt.cm.hot, origin = 'lower')
        plt.colorbar() 
        L = self.L

        # avi_traj1 = antiv_traj_arr[0] # the trajecory of the ith antivortex 
        # v1 = plt.scatter(avi_traj1[0][0]+0.5-L/2, avi_traj1[0][1]+0.5-L/2, alpha = 0.3, s = 20, color = 'blue')

        # avi_traj2 = antiv_traj_arr[1] # the trajecory of the ith antivortex 
        # v2 = plt.scatter(avi_traj2[0][0]+0.5-L/2, avi_traj2[0][1]+0.5-L/2, alpha = 0.3, s = 20, color = 'blue')


        # try storing in an array 
        # vort_arr = [] 
        
        # for i in range(len(antiv_traj_arr)): 
        #     avi_traj = antiv_traj_arr[i] 
        #     v = plt.scatter(avi_traj[0][0]+0.5-L/2, avi_traj[0][1]+0.5-L/2, alpha = 0.3, s = 20, color = 'blue')
        #     vort_arr.append(v) 
        v1 = plt.scatter(vortex1_traj[0][1]+0.5-L/2, vortex1_traj[0][2]+0.5-L/2, alpha = 0.3, s = 20, color = 'blue')
        v2 = plt.scatter(vortex2_traj[0][1]+0.5-L/2, vortex2_traj[0][2]+0.5-L/2, alpha = 0.3, s = 20, color = 'blue')


        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
        time_text.set_text('time = 0')

        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.title(f'Animation for L={L}')

        def animate(i): 
            data.set_data(np.abs(self.psi_snaps[i])**2)
            if i < len(vortex1_traj): 
                v1.set_offsets([vortex1_traj[i][1]+0.5-L/2, vortex1_traj[i][2]+0.5-L/2])
            else: 
                v1.set_offsets([np.nan, np.nan])
            if i < len(vortex2_traj): 
                v2.set_offsets([vortex2_traj[i][1]+0.5-L/2, vortex2_traj[i][2]+0.5-L/2])
            else: 
                v2.set_offsets([np.nan, np.nan])

            # for j in range(len(vort_arr)): 
            #     vort_arr[j].set_offsets([antiv_traj_arr[j][i][0]+0.5-L/2, antiv_traj_arr[j][i][1]+0.5-L/2])
    
            #time_text.set_text('time = %.1d' % time_tracking[i]) # find an array that tracks the time or define one based on dt and the number of points 
            #return data, time_text

            vort_arr = [v1,v2]
            #return data, time_text, *vort_arr
            return data, *vort_arr
        anim = animation.FuncAnimation(fig, animate, frames = len(self.psi_snaps), blit = True)
        anim.save(path)
        print(path)
        #plt.show() 

        return anim

    # animate the circulation
    def generalAnimation(self, filename, dataset, periodic = False): 
        #time_tracking = np.arange(0, len(dataset))*250*g.gpeobj.dt
        if filename != None: 
                path = fr"C:\Users\TQC User\Desktop\BECs2\{filename}_circ.mp4"
        fig, ax = plt.subplots() 
        if not periodic: 
            data = plt.imshow(dataset[0],  extent = [-self.L/2, self.L/2, -self.L/2, self.L/2], origin = 'lower')
        else: 
            data = plt.imshow(dataset[0], extent = [-self.L/2, self.L/2, -self.L/2, self.L/2], cmap = 'twilight', origin = 'lower')
        plt.colorbar() 
        plt.clim(-2*np.pi, 2*np.pi)

        
        L = self.L
        vortex1_traj = self.vortex1
        vortex2_traj = self.vortex2 
        v1 = plt.scatter(vortex1_traj[0][1]-L/2, vortex1_traj[0][2]-L/2, alpha = 0.3, s = 20, color = 'blue')
        v2 = plt.scatter(vortex2_traj[0][1]-L/2, vortex2_traj[0][2]-L/2, alpha = 0.3, s = 20, color = 'blue')
        time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
        time_text.set_text('time = 0')

        plt.xlabel("x", fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.title(f'Animation for L={L}')

        def animate(i): 
            data.set_data(dataset[i])


        #    for j in range(len(vort_arr)): 
        #         vort_arr[j].set_offsets([antiv_traj_arr[j][i][0]+0.5-L/2, antiv_traj_arr[j][i][1]+0.5-L/2])
    
            #time_text.set_text('time = %.1d' % time_tracking[i]) # find an array that tracks the time or define one based on dt and the number of points 
            #return data, time_text

            if i < len(vortex1_traj): 
                v1.set_offsets([vortex1_traj[i][1]+0.5-L/2, vortex1_traj[i][2]+0.5-L/2])
            else: 
                v1.set_offsets([np.nan, np.nan])
            if i < len(vortex2_traj): 
                v2.set_offsets([vortex2_traj[i][1]+0.5-L/2, vortex2_traj[i][2]+0.5-L/2])
            else: 
                v2.set_offsets([np.nan, np.nan])

            #v1.set_offsets([vortex1_traj[i][1]+0.5-L/2, vortex1_traj[i][2]+0.5-L/2])
            #v2.set_offsets([vortex2_traj[i][1]+0.5-L/2, vortex2_traj[i][2]+0.5-L/2])
            vort_arr = [v1, v2]
            return data, *vort_arr
        anim = animation.FuncAnimation(fig, animate, frames = len(dataset), blit = False)
        anim.save(path)
        
        #plt.show() 

        return anim 

    def main(self): 
        vort = [] 
        avort = [] 

        for i in range(len(self.psi_snaps)): 
            detection = self.detectVortices(self.psi_snaps[i])
            vort.append(detection[0])  
            avort.append(detection[1]) 
            self.circ_frames[i] = detection[2] 

        tracks = self.track_vortices_across_frames(avort)
        self.tracks = tracks 

        # Sort: Find the two IDs that correspond to original pair 
        ids = sorted(tracks, key=lambda k: len(tracks[k]), reverse=True)[:2] # sort by the length of the tracks
        track1 = tracks[ids[0]]
        track2 = tracks[ids[1]]

        self.vortex1 = track1
        self.vortex2 = track2

        # Compute and plot distance and angles 
        self.distances, self.angles = self.compute_distance_between_tracks(track1, track2)
        if self.plot: 
            self.plot_distance_over_time(self.distances)
            self.plot_angle_over_time(self.angles)

class CompareDistances(): 
    def __init__(self, temperatures = np.arange(0.5, 1.1, step = 0.1), numSamples = 3, numRealSteps = 50000, runAnim = False, numAnimations = 1, animFileName = 'unnamed'): 
        self.temperatures = temperatures 
        self.numSamples = numSamples
        self.numRealSteps = numRealSteps 
        self.runAnim = runAnim 
        self.numAnimations = numAnimations
        self.animName = animFileName
        
        self.distances = [] 
        self.angles = [] 

        self.distances_std = [] 
        self.angles_std = [] 

        self.calcAverages() 

    def calcAverages(self): 
        self.all_distance_trajectories = np.zeros((self.numSamples, len(self.temperatures), self.numRealSteps//250+2)) 
        self.all_angle_trajectories = np.zeros((self.numSamples, len(self.temperatures), self.numRealSteps//250+2))
        for t, temp in enumerate(self.temperatures): 
            sampled_distances = []
            sampled_angles = []
            avg_dist_t = np.zeros(self.numRealSteps//250+2)
            avg_angle_t = np.zeros(self.numRealSteps//250+2)
            for i in range(self.numSamples): 
                print(i) 
                print(int(self.numSamples//self.numAnimations))
                if self.runAnim and (i+1)%int(self.numSamples//self.numAnimations) == 0:
                    print("Doing animation")
                    g = gpe(npoints = 2**6, numImagSteps = 2000, numRealSteps = self.numRealSteps, dtcoef = 0.0005, boxthickness = 0.4, Nsamples = 1, runAnim = True, animFileName=self.animName, Tfact = temp, dst = False, vortex = True)
                else: 
                    g = gpe(npoints = 2**6, numImagSteps = 2000, numRealSteps = self.numRealSteps, dtcoef = 0.0005, boxthickness = 0.4, Nsamples = 1, runAnim = False, Tfact = temp, dst = False, vortex = True)
                v = VortexTracker(g.snaps, g.L, g.dx, plot = False, animFileName = self.animName) 
                times,dist = zip(*v.distances) 
                times,angles = zip(*v.angles) 
                if len(dist) < len(avg_dist_t): 
                    dist = np.pad(dist, (0,len(avg_dist_t)-len(dist)), 'constant', constant_values = np.nan)
                    angles = np.pad(angles, (0,len(avg_angle_t)-len(angles)), 'constant', constant_values = np.nan)

                # collect the distances and angles into an array 
                sampled_distances.append(dist) 
                sampled_angles.append(angles) 
                
                self.all_distance_trajectories[i][t] = dist 
                self.all_angle_trajectories[i][t] = angles


                #avg_dist_t += np.array(dist) 
                #avg_angle_t += np.array(angles) 
            avg_dist_t = np.ma.average(np.ma.masked_array(sampled_distances, np.isnan(sampled_distances)), axis = 0)
            avg_angle_t = np.ma.average(np.ma.masked_array(sampled_angles, np.isnan(sampled_angles)), axis = 0)

            std_dist_t = np.ma.std(np.ma.masked_array(sampled_distances, np.isnan(sampled_distances)), axis = 0)
            std_angle_t = np.ma.std(np.ma.masked_array(sampled_angles, np.isnan(sampled_angles)), axis = 0)
            #avg_dist_t = np.average(sampled_distances, axis = 0) 
            #avg_angle_t = np.average(sampled_angles, axis = 0)
            print(avg_dist_t)
            #avg_dist_t = avg_dist_t/self.numSamples
            #avg_angle_t = avg_angle_t/self.numSamples 
            

            self.distances.append(avg_dist_t)
            self.angles.append(avg_angle_t) 

            self.distances_std.append(std_dist_t) 
            self.angles_std.append(std_angle_t)

        self.times = times 
        self.dt = g.gpeobj.dt








