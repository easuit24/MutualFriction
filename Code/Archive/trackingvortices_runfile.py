# this file runs the analysis and tracking algorithm for vortices 

# imports 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from itertools import combinations 
import argparse 

from PointTracking_v2 import PointTracker as pt
from classicalfield_orig import FiniteTempGPE as gpe

def animatepsi(filename): 
    '''
    Animates the density distribution for the 
    '''
    time_tracking = np.arange(0, len(g.snaps))*250*g.gpeobj.dt
    if filename != None: 
            path = fr"C:\Users\TQC User\Desktop\BECs2\{filename}.mp4"
    fig, ax = plt.subplots() 
    data = plt.imshow(np.abs(g.snaps[0])**2, extent = [-g.winL/2, g.winL/2, -g.winL/2, g.winL/2], cmap = plt.cm.hot, origin = 'lower')
    plt.colorbar() 
    L = g.L 
    vort_arr = [] 
    
    for i in range(len(antiv_traj_arr)): 
         avi_traj = antiv_traj_arr[i] 
         v = plt.scatter(avi_traj[0][0]+0.5-L/2, avi_traj[0][1]+0.5-L/2, alpha = 0.3, s = 20, color = 'blue')
         vort_arr.append(v) 


    time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
    time_text.set_text('time = 0')

    plt.xlabel("x", fontsize = 16)
    plt.ylabel('y', fontsize = 16)
    plt.title(f'Animation for L={L}')

    def animate(i): 
        data.set_data(np.abs(g.snaps[i])**2)

        for j in range(len(vort_arr)): 
             vort_arr[j].set_offsets([antiv_traj_arr[j][i][0]+0.5-L/2, antiv_traj_arr[j][i][1]+0.5-L/2])
 
        time_text.set_text('time = %.1d' % time_tracking[i])  

        return data, time_text, *vort_arr
    anim = animation.FuncAnimation(fig, animate, frames = len(g.snaps), blit = True)
    anim.save(path)
    #plt.show() 

    return anim 
    

def generalAnimation(filename, dataset, periodic = False): 
    time_tracking = np.arange(0, len(dataset))*250*g.gpeobj.dt
    if filename != None: 
            path = fr"C:\Users\TQC User\Desktop\BECs2\{filename}.mp4"
    fig, ax = plt.subplots() 
    if not periodic: 
        data = plt.imshow(dataset[0],  extent = [-g.winL/2, g.winL/2, -g.winL/2, g.winL/2], origin = 'lower')
    else: 
        data = plt.imshow(dataset[0], extent = [-g.winL/2, g.winL/2, -g.winL/2, g.winL/2], cmap = 'twilight', origin = 'lower')
    plt.colorbar() 
    plt.clim(-2*np.pi, 2*np.pi)

     
    L = g.L
    time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,  bbox=dict(facecolor='red', alpha=0.5))
    time_text.set_text('time = 0')

    plt.xlabel("x", fontsize = 16)
    plt.ylabel('y', fontsize = 16)
    plt.title(f'Animation for L={L}')

    def animate(i): 
        data.set_data(dataset[i])


     #    for j in range(len(vort_arr)): 
     #         vort_arr[j].set_offsets([antiv_traj_arr[j][i][0]+0.5-L/2, antiv_traj_arr[j][i][1]+0.5-L/2])
 
        time_text.set_text('time = %.1d' % time_tracking[i]) # find an array that tracks the time or define one based on dt and the number of points 
        #return data, time_text

        #vort_arr = [v1,v2]
        return data, time_text
    anim = animation.FuncAnimation(fig, animate, frames = len(dataset), blit = True)
    anim.save(path)
    
    plt.show() 

    return anim 
    

parser = argparse.ArgumentParser(description = 'Vortex Tracking Argument Parser')

parser.add_argument(
    "-T",
    "--temp",
    default = 0.5, 
    help = 'Temperature argument',
    type = float
)
parser.add_argument(
     "-d",
     "--debug",
     default = False, 
     help = 'Whether to run in debug mode',
     type = bool 
)
parser.add_argument(
     "-nR",
     "--numReal",
     default = 250000,
     help = 'Number of Real Steps',
     type = int 
)

args = parser.parse_args() 
tempFact = args.temp 
debug = args.debug 
numRealSteps = args.numReal

print(args)
g = gpe(npoints = 2**6, numImagSteps = 2000, numRealSteps = numRealSteps, dtcoef = 0.0005, boxthickness = 0.4, Nsamples = 1, runAnim = True, animFileName = 'test.mp4', Tfact = tempFact, dst = False, vortex = True)

tracker = pt(g.snaps, g.dx, g.L, g.gpeobj.dt)

v_traj, antiv_traj, circ_array = pt.labelVortices(tracker, getCirc = True) 

# convert the lists into arrays 

v_traj_arr = np.array(v_traj) 
antiv_traj_arr = np.array(antiv_traj) 

# distance 
vortex_combos = np.array(list(combinations(antiv_traj_arr, 2))) # specifically for the antivortex case.... lacking for the vortex/antivortex pair case... 

distance_arr = np.zeros((len(vortex_combos), len(g.snaps)))
angle_arr = np.zeros((len(vortex_combos), len(g.snaps))) 

for i, pair in enumerate(vortex_combos): # for each pair - find the distance 
    for j in range(len(vortex_combos[0][0])): 
 
        point1 = pair[0][j] 
        point2 = pair[1][j] 
        xsep = point1[0] - point2[0] 
        ysep = point1[1] - point2[1] 
        dist = np.sqrt(np.abs(point1[0] - point2[0])**2 + np.abs(point1[1] - point2[1])**2) 
        ang = np.arctan(ysep/xsep) 

        distance_arr[i,j] = dist 
        angle_arr[i,j] = ang

# figure for distance 
plt.figure() 
for i in range(len(distance_arr)): 
    plt.plot(np.linspace(0,len(distance_arr[i]), len(distance_arr[i])), distance_arr[i])
plt.xlabel('Time Step')
plt.ylabel('Distance') 
plt.title('Distance between vortices')
# save
plt.savefig(f'distanceplot_{tempFact}.png')
np.savetxt(f'distancetraj_{tempFact}.csv', distance_arr)

# figure for angle 
plt.figure() 
for i in range(len(angle_arr)): 
    plt.plot(np.linspace(0,len(angle_arr[i]), len(angle_arr[i])), angle_arr[i])
plt.xlabel('Time Step')
plt.ylabel('Angle [radians]') 
plt.title('Angle between vortices')
# save
plt.savefig(f'angleplot_{tempFact}.png')

# animate the density itself
animatepsi(f'densityanim_{tempFact}.mp4')

if debug: 
    # figure for circulation
    generalAnimation(f'circanim_{tempFact}.mp4', circ_array)

    # Animating the Phase 
    phase = [] 
    for snap in range(len(g.snaps)): 
        phase.append(np.angle(g.snaps[snap][int(g.L/2/g.dx):int(3*g.L/2/g.dx), int(g.L/2/g.dx): int(3*g.L/2/g.dx)]))
    phase_arr = np.array(phase)

    generalAnimation(f'phaseanim_{tempFact}.mp4', phase_arr, periodic = True)
