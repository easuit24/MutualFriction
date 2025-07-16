import numpy as np
import matplotlib.pyplot as plt

from gpetimeev2 import GPETimeEv as gpe

g = gpe(dim = 2) 


# figure of the output from the imaginary time evolution 
plt.figure() 
plt.imshow(np.real(g.psi))
plt.colorbar()
plt.savefig(f'imagevresult_{g.L}.png') 

# animation output of the total evolution 
g.animatepsi2d_save()

# potential energy plot 
plt.figure() 
plt.plot(g.Ep_arr) 
plt.xlabel("Time Step")
plt.ylabel("Potential Energy")
plt.savefig(f'potenresult_{g.w}{g.L}')

# virial check 
plt.figure()
plt.plot(g.virial) 
plt.xlabel("Time Step")
plt.ylabel("-2KE + 2PE - 3INT")
plt.savefig(f'virialresult_{g.w}{g.L}')

