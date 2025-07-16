import numpy as np
import matplotlib.pyplot as plt
import os 

from gpetimeev2 import GPETimeEv as gpe

print("Running Script")
g = gpe(dim = 2) 

os.chdir("Outputs")
os.mkdir(f'Results_w{g.w}L{g.L}_5')
os.chdir(f'Results_w{g.w}L{g.L}_5')

# generate images

print("Output")

# animation output 
g.animatepsi2d_save()

# end wavefunction with contours 
plt.figure()
plt.imshow(np.abs(g.dynpsi), extent = [-g.winL/2, g.winL/2, -g.winL/2, g.winL/2], cmap = plt.cm.hot) 
plt.colorbar() 
plt.contour(g.xi[0], g.xi[1], g.Vs)
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(f'wfimg_w{g.w}L{g.L}.png')

# energy plot 
plt.figure() 
plt.plot(g.Ei_arr, label = "Interaction Energy")
plt.plot(g.Ek_arr, label = "Kinetic Energy")
plt.plot(g.Ep_arr, label = "Potential Energy")
plt.xlabel("Time Step * 500")
plt.ylabel("Energy")
plt.legend()
plt.savefig(f'energyplot_w{g.w}L{g.L}.png')

# total energy plot 
plt.figure() 
total_en = np.array(g.Ek_arr) + np.array(g.Ep_arr) + np.array(g.Ei_arr)
plt.xlabel("Time Step * 500")
plt.ylabel("Total Energy")
plt.plot(total_en)
plt.savefig(f'totalen_w{g.w}L{g.L}.png')

# virial theorem plot 
plt.figure() 
virial = -2*np.array(g.Ek_arr) + 2*np.array(g.Ep_arr) - 3* np.array(g.Ei_arr)
plt.plot(virial) 
plt.xlabel("Time Step * 500")
plt.ylabel("-2*KE + 2*PE - 3*INT")
plt.savefig(f'virialtest_w{g.w}L{g.L}.png')









