import matplotlib.pyplot as plt
import numpy as np

q_table = np.load("qtables/cap_basic.npy")

plt.imshow(q_table, cmap = 'hot',origin='lower',aspect = 'auto')
plt.xlabel("Price, £")
plt.ylabel("Current number of spaces taken")
plt.title("Q-Value of finite capacity")
plt.colorbar(label="Q-Value")
plt.savefig("plots/cap_basic.pdf")
plt.savefig("plots/cap_basic.png")