"""
It is not as straightforward to render plots and graphs from eagle.

Steps needed:
1) When connecting to eagle via ssh we need to somehow provide the -X flag
  (X forwarding).  Apparently, as of April 2019, pycharm does not provide this
  option. However, there does appear to be a way to make it happen:
  https://stackoverflow.com/questions/41892039/how-to-enable-x11-forwarding-
  in-pycharm-ssh-session.
2) We need to change the "X11Forwarding" value to "yes" in eagle's sshd_config
  file. This just requires 5 seconds of admin privileges.
3) Use/install and use a back-end that is compatible with x11 forwarding.
   Not sure which, but matplotlib will let us know when we start trying.

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# check backend
matplotlib.get_backend()
matplotlib.use('Qt5Agg')


# Trying to use pycharm plotting functionality through a console in eagle
r = np.random.normal(100, 40, 10000)
t = r.reshape((100, 100))

# Image?
plt.imshow(t)
plt.show()

