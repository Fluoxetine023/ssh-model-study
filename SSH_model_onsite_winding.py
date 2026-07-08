from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Parameters
# -----------------------
v = 0.2
w = 1.0
onsite_values = [0.1, 0.0, -0.1]

k = np.linspace(-np.pi, np.pi, 800)

dx = v + w*np.cos(k)
dy = w*np.sin(k)

# -----------------------
# Figure style
# -----------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 1.2
})

fig = plt.figure(figsize=(18,6))

for i,u in enumerate(onsite_values):

    ax = fig.add_subplot(1,3,i+1, projection='3d')

    dz = np.full_like(k,u)

    # Main trajectory
    ax.plot(dx,dy,dz,
            color='black',
            linewidth=2.5)

    # Projection onto d_x-d_y plane
    ax.plot(dx,dy,np.zeros_like(k),
            linestyle='--',
            color='gray',
            linewidth=1.5)

    # Origin
    ax.scatter([0],[0],[0],
               color='black',
               s=25)

    # Transparent plane (z=0)
    xx = np.linspace(-1.2,1.2,2)
    yy = np.linspace(-1.2,1.2,2)
    XX,YY = np.meshgrid(xx,yy)
    ZZ = np.zeros_like(XX)

    ax.plot_surface(
        XX,YY,ZZ,
        alpha=0.15,
        color='lightgray',
        edgecolor='none'
    )

    # Limits
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-0.25, 0.25)

    ax.set_box_aspect((1,1,0.45))

    # Viewing angle
    ax.view_init(elev=25, azim=-60)

    ax.set_xlabel(r'$d_x$', labelpad=8)
    ax.set_ylabel(r'$d_y$', labelpad=8)
    ax.set_zlabel(r'$d_z$', labelpad=8)

    ax.set_title(rf'$u={u}$', fontsize=25)

    ax.grid(False)

fig.subplots_adjust(
    left=0.04,
    right=0.96,
    bottom=0.10,
    top=0.90,
    wspace=0.18
)

plt.savefig(
    "onsite_dvector_3D.pdf",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.2
)
plt.show()