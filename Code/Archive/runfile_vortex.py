import argparse 
import time

from gpevortex import GPETimeEv as gpe

parser = argparse.ArgumentParser(description = 'Vortex Parameter Arugments')

parser.add_argument(
    '-is',
    '--imagSteps',
    default = 2000,
    help = 'number of imaginary steps (int)',
    type = int
)
parser.add_argument(
    "-rs",
    "--realSteps",
    default = 10000,
    help = 'number of real steps (int)',
    type = int
)
parser.add_argument(
    '-av',
    '--antivortex',
    default = False,
    help = 'presence of antivortex (bool)',
    type = bool
)
parser.add_argument(
    '-d',
    '--dist',
    default = 1,
    help = 'distance between pairs (float)',
    type = float
)
parser.add_argument(
    '-L',
    '--length',
    default = 20,
    help = 'length of a side of the box potential (int)',
    type = int
)
parser.add_argument(
    '-tc',
    '--timecoef',
    help = 'time coefficient for dt',
    default = 0.1,
    type = float
)
parser.add_argument(
    '-s',
    '--spawn',
    default = 'pair',
    help = 'tpye of vortex behavior to spawn',
    type = str
)
parser.add_argument(
    '-f',
    '--filename',
    help = 'file name output for animation',
    type = str
)
args = parser.parse_args() 
print(args) 
imagSteps = args.imagSteps
realSteps = args.realSteps
antiVortex = args.antivortex
dist = args.dist
length = args.length
timecoef = args.timecoef
spawnTyp = args.spawn
filename = args.filename 

t0 = time.time() 
g = gpe(dim = 2, L = length, dtcoef = timecoef, numImagSteps=imagSteps, numRealSteps=realSteps, antiV = antiVortex, dist = dist, spawnType=spawnTyp ) 
print("Run Time: ", time.time() - t0) 

g.animatepsi2d_save(filename) 