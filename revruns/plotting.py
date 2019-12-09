import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# check backend
matplotlib.get_backend()
matplotlib.use('Agg')


# Trying to use pycharm plotting functionality through a console in eagle
r = np.random.normal(100, 40, 10000)
t = r.reshape((100, 100))

# Image?
plt.imshow(t)
plt.show()
