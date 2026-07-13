\
"""
Bulk SSH Band Structure and Winding Trajectories
================================================

Author
------
Vijay Vedanth Vemula

Description
-----------
This script illustrates the bulk properties of the Su–Schrieffer–Heeger
(SSH) model by plotting

- The bulk energy dispersion relation.
- The corresponding winding trajectories of the d-vector.

Five different hopping parameter sets are considered to demonstrate the
transition between the trivial, critical and topological phases.

Summer Research Internship 2026
Department of Physics
IIT (BHU), Varanasi
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 22,
    "axes.labelsize": 23,
    "axes.titlesize": 22,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

# ============================================================
# Parameters
# ============================================================

PARAMETERS = [
    (1.0, 0.0),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.0, 1.0),
]

k = np.linspace(-np.pi, np.pi, 1000)


def ssh_dispersion(v: float, w: float, k: np.ndarray) -> np.ndarray:
    """
    Compute the bulk energy dispersion relation.

    E(k) = ±√(v² + w² + 2vw cos k)
    """
    return np.sqrt(v**2 + w**2 + 2 * v * w * np.cos(k))


def winding_curve(v: float, w: float, k: np.ndarray):
    """
    Compute the d-vector trajectory

        d(k) = (v + w cos k, w sin k).
    """
    dx = v + w * np.cos(k)
    dy = w * np.sin(k)
    return dx, dy


def draw_arrow(ax, x, y, target_angle=np.pi / 4, step=8):
    """Draw an arrow indicating increasing crystal momentum."""

    center_x = np.mean(x)
    center_y = np.mean(y)

    theta = np.arctan2(y - center_y, x - center_x)

    index = np.argmin(
        np.abs(np.angle(np.exp(1j * (theta - target_angle))))
    )

    i1 = max(index - step, 0)
    i2 = min(index + step, len(x) - 1)

    ax.annotate(
        "",
        xy=(x[i2], y[i2]),
        xytext=(x[i1], y[i1]),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.8,
            color="black",
            mutation_scale=16,
        ),
    )


def plot_band(ax, v, w):
    """Plot the positive and negative bulk bands."""

    energy = ssh_dispersion(v, w, k)

    ax.plot(k, energy, lw=2)
    ax.plot(k, -energy, lw=2)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-1, 0, 1])

    ax.grid(alpha=0.25)
    ax.set_title(rf"$v={v:.1f},\; w={w:.1f}$")


def plot_winding(ax, v, w):
    """Plot the winding trajectory in the d_x-d_y plane."""

    dx, dy = winding_curve(v, w, k)

    ax.plot(dx, dy, lw=2)

    draw_arrow(ax, dx, dy)

    ax.scatter(0, 0, color="black", s=22, zorder=5)

    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(0, color="0.75", lw=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xticks([-1.5, 0, 1.5])
    ax.set_yticks([-1.5, 0, 1.5])

    ax.grid(alpha=0.20)


def main():
    """Generate the band structure and winding trajectory figure."""

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 8),
        sharey="row",
        constrained_layout=True,
    )

    for column, (v, w) in enumerate(PARAMETERS):
        plot_band(axes[0, column], v, w)
        plot_winding(axes[1, column], v, w)

        axes[0, column].set_xlabel(r"$k$")
        axes[1, column].set_xlabel(r"$d_x$")

    axes[0, 0].set_ylabel(r"$E(k)$")
    axes[1, 0].set_ylabel(r"$d_y$")

    plt.savefig("ssh_bands_winding.pdf", bbox_inches="tight")
    plt.savefig("ssh_bands_winding.png", dpi=600, bbox_inches="tight")

    print("Saved: ssh_bands_winding.pdf")
    print("Saved: ssh_bands_winding.png")

    plt.show()


if __name__ == "__main__":
    main()
