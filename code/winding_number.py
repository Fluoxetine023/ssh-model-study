"""
Three-Dimensional Winding Trajectories of the SSH Model
=======================================================

Author
------
Vijay Vedanth Vemula

Description
-----------
This script visualizes the three-dimensional d-vector trajectory of the
Su-Schrieffer-Heeger (SSH) model after introducing a staggered on-site
potential.

For each value of the on-site potential u, the vector

    d(k) = (v + w cos(k), w sin(k), u)

is plotted as the crystal momentum k traverses the first Brillouin zone.

The figure illustrates how introducing a non-zero on-site potential
displaces the trajectory out of the d_x-d_y plane, thereby breaking
chiral symmetry and eliminating the planar winding number.

The generated figure is saved as

    onsite_dvector_3D.pdf

Summer Research Internship 2026
Department of Physics
IIT (BHU), Varanasi
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = Path("Report/Figures") / "onsite_potential"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Plotting parameters
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 1.2,
})

# ============================================================
# Parameters
# ============================================================

v = 0.2                  # Intracell hopping
w = 1.0                  # Intercell hopping
onsite_values = [0.1, 0.0, -0.1]

# Crystal momentum
k = np.linspace(-np.pi, np.pi, 800)

# Components of the d-vector:
#
# H(k) = d(k) · σ
#
# where
# d(k) = (v + w cos(k), w sin(k), u)

dx = v + w * np.cos(k)
dy = w * np.sin(k)


def main():
    """Generate the 3D d-vector trajectories for different on-site potentials."""

    fig = plt.figure(figsize=(18, 6))

    for i, u in enumerate(onsite_values):

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # The staggered on-site potential introduces
        # a constant d_z component.
        dz = np.full_like(k, u)

        # ----------------------------------------------------
        # Main trajectory
        # ----------------------------------------------------

        ax.plot(
            dx,
            dy,
            dz,
            color="black",
            linewidth=2.5,
        )

        # ----------------------------------------------------
        # Projection onto the d_x-d_y plane
        # ----------------------------------------------------

        ax.plot(
            dx,
            dy,
            np.zeros_like(k),
            linestyle="--",
            color="gray",
            linewidth=1.5,
        )

        # ----------------------------------------------------
        # Origin
        # ----------------------------------------------------

        ax.scatter(
            [0],
            [0],
            [0],
            color="black",
            s=25,
        )

        # ----------------------------------------------------
        # Transparent reference plane (d_z = 0)
        # ----------------------------------------------------

        xx = np.linspace(-1.2, 1.2, 2)
        yy = np.linspace(-1.2, 1.2, 2)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.zeros_like(XX)

        ax.plot_surface(
            XX,
            YY,
            ZZ,
            alpha=0.15,
            color="lightgray",
            edgecolor="none",
        )

        # ----------------------------------------------------
        # Axes limits and aspect ratio
        # ----------------------------------------------------

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_zlim(-0.25, 0.25)

        ax.set_box_aspect((1, 1, 0.45))

        # Viewing angle
        ax.view_init(elev=25, azim=-60)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(r"$d_x$", labelpad=8)
        ax.set_ylabel(r"$d_y$", labelpad=8)
        ax.set_zlabel(r"$d_z$", labelpad=8)

        ax.set_title(rf"$u={u}$", fontsize=25)

        ax.grid(False)

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.10,
        top=0.90,
        wspace=0.18,
    )

    filename = OUTPUT_DIR / "onsite_dvector_3D.pdf"

    plt.savefig(
        filename,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    print(f"Saved: {filename}")

    plt.show()


if __name__ == "__main__":
    main()