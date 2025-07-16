import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from classicalfield_orig import FiniteTempGPE as gpe

from scipy.fft import fft2, ifft2, fftshift, ifftshift 
from scipy.optimize import curve_fit
import argparse 
import time 

print('main')
parser = argparse.ArgumentParser()
parser.add_argument(
    '-nI',
    '--numImag',
    default = 2000, 
    help = 'Number of Imaginary Steps (int)',
    type = int
)
parser.add_argument(
    '-nR',
    '--numReal',
    default = 1000,
    help = 'Number of Real Steps (int)',
    type = int
)
parser.add_argument(
    '-dt',
    '--dtcoef',
    default = 0.0025,
    help = 'dt coefficient (float)',
    type = float
)
parser.add_argument(
    '-N',
    '--Nsamp',
    default = 1,
    help = 'Number of Samples',
    type = int 
)
parser.add_argument(
    '-vo',
    '--vortex',
    default = False,
    help = 'Vortex appearance (boolean)',
    type = bool
)
parser.add_argument(
    '-T',
    '--Tfact',
    default = 1/2,
    help = 'Temperature coefficient (float)',
    type = float
)
parser.add_argument(
    '-im',
    '--imp',
    default = False, 
    help = 'Presence of imported wavefunction array',
    type = bool 
)
parser.add_argument(
    '-ip',
    '--impPsi',
    default = 'None',
    help = 'Imported wavefunction array file name',
    type = str
)
parser.add_argument(
    '-w',
    '--winMult',
    default = 2, 
    help = 'Window multiplier (int)',
    type = int 
)
parser.add_argument(
    '-ra',
    '--runAnim',
    default = False, 
    help = 'Whether to display the animation',
    type = bool 
)
parser.add_argument(
    '-a',
    '--animName',
    default = 'None', 
    help = 'File name for animation .mp4 file',
    type = str 
)
parser.add_argument(
    '-np',
    '--numpoints',
    default = 2**7, 
    help = 'Number of points along axis',
    type = int
)
parser.add_argument(
    '-bt',
    '--boxthickness',
    default = 3, 
    help = 'Thickness of the potential wall i.e. how gradual the incline is',
    type = float
)
args = parser.parse_args() 
print(args)
numImag = args.numImag 
numReal = args.numReal 
dtcoef = args.dtcoef 
Nsamples = args.Nsamp 
vortex = args.vortex
Tfact = args.Tfact 
imp = args.imp
npoints = args.numpoints
boxthickness = args.boxthickness 
winMult = args.winMult 
if imp:
    filename = args.impPsi 
    impPsi = np.loadtxt(filename, dtype = np.complex_)
else: 
    impPsi = None 
runAnim = args.runAnim
if runAnim: 
    animName = args.animName 
else: 
    animName = None



t0 = time.time() 

g = gpe(npoints = npoints, numImagSteps=numImag, numRealSteps = numReal, boxthickness=boxthickness, winMult = winMult, dtcoef = dtcoef, Nsamples=Nsamples, vortex = vortex, Tfact =Tfact, imp = imp, impPsi = impPsi, runAnim = runAnim, animFileName = animName, dst = False) 
print("Total Run Time: ", time.time() - t0)
if runAnim: 
    
    plt.figure() 
    plt.plot(np.abs(g.snaps[0][len(g.snaps[0])//2])**2, label = 'Initial State')
    plt.plot(np.abs(g.snaps[-1][len(g.snaps[0])//2])**2, linestyle = '--', label = 'Final State')
    plt.xlabel('x')
    plt.ylabel('Wavefunction Density')
    plt.title('Slice of Position Wavefunction Density with Vortices')
    plt.legend() 
    plt.savefig(f'{animName}_dens_distribution.png')
    plt.show()


    init_k_states = [] 

    grid = g.extractBox(g.xi, g.wf_samples[0])[2]
    for i in range(len(g.wf_samples)): 
        init_k_states.append(g.extractBox(g.xi, g.wf_samples[i])[3])

    avg_initk = np.mean(np.abs(init_k_states)**2, axis = 0, dtype = np.complex_)
    avg_finalk = np.mean(np.abs(g.short_wfk)**2, axis = 0, dtype = np.complex_)

    plt.figure() 
    plt.plot(fftshift(grid), fftshift(avg_initk[0]), label = 'Initial State')
    plt.plot(fftshift(grid), fftshift(avg_finalk[0]), label = 'Final State')
    plt.yscale('log') 
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('Momentum Density')
    plt.title('Slice of Momentum Wavefunction Density with Vortices')
    plt.legend() 
    plt.savefig(f'{animName}_momentum_distribution.png')
    plt.show() 
    #np.savetxt(animName, np.reshape(g.snaps, -1))
    np.savetxt('grid.csv', grid)
    np.savetxt('init_k.csv', avg_initk)
    np.savetxt('final_k.csv', avg_finalk)


