import matplotlib.pyplot as plt
import numpy as np

q_table = np.load("qtables/LoS.npy")

plt.imshow(q_table, cmap='hot',origin='lower')
plt.xlabel("Day 2 Price, £")
plt.ylabel("Day 1 Price, £")
plt.title("Q-Values of infinite capacity, booking with length of stay")
plt.colorbar(label = 'Q-Value')
plt.savefig("plots/LoS.pdf")
plt.savefig("plots/LoS.png")