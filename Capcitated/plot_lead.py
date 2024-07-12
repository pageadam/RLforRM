import matplotlib.pyplot as plt
import numpy as np

q_table = np.load("qtables/cap_lead.npy")
figure, axis = plt.subplots(2, 4) 
axis[0,0].imshow(q_table[0,:,:], cmap='hot',origin='lower')
axis[0,1].imshow(q_table[1,:,:], cmap='hot',origin='lower')
axis[0,2].imshow(q_table[2,:,:], cmap='hot',origin='lower')
axis[0,3].imshow(q_table[3,:,:], cmap='hot',origin='lower')
axis[1,0].imshow(q_table[4,:,:], cmap='hot',origin='lower')
axis[1,1].imshow(q_table[5,:,:], cmap='hot',origin='lower')
axis[1,2].imshow(q_table[6,:,:], cmap='hot',origin='lower')
axis[1,3].imshow(q_table[7,:,:], cmap='hot',origin='lower')
plt.show()
plt.savefig("plots/cap_lead.pdf")
plt.savefig("plots/cap_lead.png")