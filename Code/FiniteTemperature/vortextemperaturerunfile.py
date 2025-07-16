import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse 
import time 

from classicalfield_orig import FiniteTempGPE as gpe 
#from VortexLabeling import VortexTracker as vt 
from VortexLabeling import CompareDistances as comp
#from PointTracking_v2 import PointTracker as pt

t0 = time.time() 
parser = argparse.ArgumentParser(description = 'Vortex temperature analysis arguments')

parser.add_argument(
    '-nR',
    '--numRealSteps',
    default = 50000,
    help = 'Number of real steps for simulation',
    type = int
)
parser.add_argument(
    '-ns',
    '--numSamples',
    default = 20,
    help = 'Number of samples for each temperature',
    type = int
)
parser.add_argument(
    '-s',
    '--start',
    default = 0,
    help = 'Starting temperature for simulation',
    type = float 
)
parser.add_argument(
    '-e',
    '--end',
    default = 1.1,
    help = 'End temperature for simulation',
    type = float 
)
parser.add_argument(
    '-st',
    '--step',
    default = 0.2,
    help = 'step size for temperature increments',
    type = float 
)
parser.add_argument(
    '-f',
    '--filename',
    default = 'unnamed',
    help = 'filename for the distance .csv file output',
    type = str 
)
parser.add_argument(
    '-r',
    '--runAnim',
    default = False,
    help = 'whether to run the animation or not (boolean)',
    type = bool 
)
args = parser.parse_args() 
print("args: ", args)
numRealSteps = args.numRealSteps 
numSamples = args.numSamples 
temperatures = np.arange(args.start, args.end, args.step)
fileinfo = f'T{args.start}_S{args.numSamples}_s{args.numRealSteps}_{np.round(time.time(), 3)}'
c = comp(numRealSteps = numRealSteps, numSamples = numSamples, temperatures = temperatures, runAnim = args.runAnim, animFileName=args.filename)

# save the distance trajectory 
np.savetxt(f'{args.filename}.csv', c.distances)
np.savetxt(f'{args.filename}_alltraj.csv', c.all_distance_trajectories[:,0])
np.savetxt(f'{args.filename}_alltraj_ang.csv', c.all_angle_trajectories[:,0])

plt.figure() 
for i in range(len(c.distances)): 
    plt.plot( c.distances[i], label = c.temperatures[i], marker = 'o')
plt.grid(True)
plt.legend() 
plt.xlabel("Frame")
plt.ylabel("Distance")
plt.title("Distance Between Vortices Over Time")
plt.savefig(f'distances_range{args.start}{args.end}_{args.filename}.png')

plt.figure() 
for i in range(len(c.distances)): 
    plt.errorbar(np.arange(0, len(c.distances[i])), c.distances[i], c.distances_std, marker = 'o')
plt.grid(True)
plt.xlabel("Frame")
plt.ylabel("Distance")
plt.title("Distance Between Vortices Over Time")
plt.savefig(f'distances_range{args.start}{args.end}_{args.filename}_errorplot.png')

print("Total Run Time: ", time.time() - t0)
#plt.show() 