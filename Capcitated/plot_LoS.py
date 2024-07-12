import matplotlib.pyplot as plt
import numpy as np

q_table = np.load("qtables/cap_LoS.npy")


plt.imshow(q_table[0,0,:,:], cmap='hot',origin='lower',vmin=0)
plt.xlabel("Day 2 Price, £")
plt.ylabel("Day 1 Price, £")
plt.colorbar(label = 'Q-value')
plt.title('Q-Learner values for C = [0,0]')
plt.show()
plt.savefig("plots/cap_LoS.pdf")
plt.savefig("plots/cap_LoS.png")