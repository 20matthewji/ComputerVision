#!/usr/local/bin/python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x_side = np.arange(-5, 5, 0.04)
y_side = np.arange(-5, 5, 0.04)
np.meshgrid(x_side, y_side)

Z = np.exp(-(X**2+Y**2)) + 2*np.exp(-((X-2)**2 + Y**2))

COLORS = np.empty(X.shape, dtype=str)
COLORS[:, :] ='b'

# 2D simple plot:
ax=pylab.subplot(111)
ax.set_xbound(0, 5)
ax.set_ybound(-5, 0)
ax.set_title("Plot Data")
ax.set_xlabel("x")
ax.set_ylabel("y")
trajectory = ax.plot(X,Y, linestyle='None', marker='o', color='b')

# 3D surface plot
fig=plt.figure()
ax=fig.gca(projection='3d')
res=np.array([X,Y,Z])
# surf = ax.plot_surface(X,Y,Z, facecolors=COLORS, rstride=1, cstride=1, linewidth=0)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.view_init(elev=25, azim=-120)
plt.show()
