import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('unsw-nb15/versions/1/UNSW_NB15_training-set.csv')
data = np.array([df['spkts'], df['sbytes'], df['ct_dst_ltm']])  # Example columns, adjust as needed

plt.scatter(data[0], data[1], label='Data Points')
plt.xlabel('Duration')
plt.ylabel('Rate')
plt.legend()
plt.show()