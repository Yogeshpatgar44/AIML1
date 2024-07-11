import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.3* np.random.randn(100)

smooth_data = lowess (y, x, frac=0.25)
plt.plot(x, y, 'r.', label='Original Data')
plt.plot(smooth_data[:, 0], smooth_data[:, 1], 'b', label='LOWESS Fit')
plt.legend()
plt.show()
