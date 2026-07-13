\
"""
Finite SSH Chain with On-Site Potential
=======================================

Author
------
Vijay Vedanth Vemula

Description
-----------
This script extends the Su–Schrieffer–Heeger (SSH) model by introducing a
staggered on-site potential (+u on sublattice A and -u on sublattice B).

For each parameter set, the script:

- Constructs the finite SSH Hamiltonian.
- Numerically diagonalizes the Hamiltonian.
- Identifies the positive- and negative-energy edge states.
- Plots their probability distributions.
- Plots the complete energy spectrum.
- Saves all figures in Report/Figures/onsite_potential/.

Summer Research Internship 2026
Department of Physics
IIT (BHU), Varanasi
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = Path("Report/Figures") / "onsite_potential"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 23,
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


def build_ssh_hamiltonian(v: float, w: float, u: float, n_cells: int) -> np.ndarray:
    """
    Construct the finite SSH Hamiltonian with a staggered on-site potential.

    Parameters
    ----------
    v : float
        Intracell hopping.
    w : float
        Intercell hopping.
    u : float
        On-site potential (+u on A, -u on B).
    n_cells : int
        Number of unit cells.

    Returns
    -------
    numpy.ndarray
        Hamiltonian matrix of dimension (2N x 2N).
    """
    dim = 2 * n_cells
    H = np.zeros((dim, dim))

    for m in range(n_cells):
        A = 2 * m
        B = A + 1

        H[A, B] = v
        H[B, A] = v

        if m < n_cells - 1:
            H[B, A + 2] = w
            H[A + 2, B] = w

        H[A, A] = u
        H[B, B] = -u

    return H


def solve_ssh_system(H: np.ndarray):
    """
    Diagonalize the Hamiltonian and extract the positive- and
    negative-energy edge states.
    """
    eigvals, eigvecs = la.eigh(H)

    positive_idx = np.where(eigvals > 1e-30)[0][0]
    negative_idx = np.where(eigvals < -1e-30)[0][-1]

    psi_pos = eigvecs[:, positive_idx]
    psi_neg = eigvecs[:, negative_idx]

    prob_pos = np.abs(psi_pos) ** 2
    prob_neg = np.abs(psi_neg) ** 2

    return (
        eigvals,
        prob_pos[0::2],
        prob_pos[1::2],
        prob_neg[0::2],
        prob_neg[1::2],
    )


def plot_probability_distribution(
    prob_A_pos,
    prob_B_pos,
    prob_A_neg,
    prob_B_neg,
    v,
    w,
    u,
):
    """Plot edge-state probability distributions."""

    n_cells = len(prob_A_pos)
    cells = np.arange(n_cells)
    width = 0.4

    if np.isclose(u, 0.0):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

        ax.bar(cells - width / 2, prob_A_pos, width, label="A")
        ax.bar(cells + width / 2, prob_B_pos, width, label="B")

        ax.set_title("Edge State")
        ax.set_ylabel(r"Probability Density $|\psi|^2$")
        ax.set_xlabel("Unit Cell Index")
        ax.set_xticks(cells)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100, sharey=True)

        datasets = [
            (prob_A_pos, prob_B_pos, "Positive Edge State"),
            (prob_A_neg, prob_B_neg, "Negative Edge State"),
        ]

        for ax, (pA, pB, title) in zip(axes, datasets):
            ax.bar(cells - width / 2, pA, width, label="A")
            ax.bar(cells + width / 2, pB, width, label="B")
            ax.set_title(title)
            ax.set_xlabel("Unit Cell Index")
            ax.set_xticks(cells)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.legend()

        axes[0].set_ylabel(r"Probability Density $|\psi|^2$")

    plt.tight_layout()

    filename = OUTPUT_DIR / f"edge_states_v={v}_w={w}_u={u}_N={n_cells}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


def plot_eigenvalue_spectrum(eigvals, v, w, u):
    """Plot the complete eigenvalue spectrum."""

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    indices = np.arange(len(eigvals))

    ax.scatter(indices, eigvals, color="#2ca02c", edgecolor="black", s=45)

    pos = np.where(eigvals > 1e-30)[0][0]
    neg = np.where(eigvals < -1e-30)[0][-1]

    ax.scatter(pos, eigvals[pos], color="red", edgecolor="black", s=90, zorder=5)
    ax.scatter(neg, eigvals[neg], color="red", edgecolor="black", s=90, zorder=5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)

    ax.set_xlabel("State Index")
    ax.set_ylabel(r"Energy $E$")
    ax.grid(True, linestyle="--", alpha=0.5)

    filename = OUTPUT_DIR / f"spectrum_v={v}_w={w}_u={u}_N={len(eigvals)//2}.pdf"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


def main():
    """Run all simulations."""

    parameter_sets = [
        (0.2, 1.0, 0.0, 10),
        (0.2, 1.0, 0.1, 10),
        (0.2, 1.0, -0.1, 10),
        (0.2, 1.0, 0.0, 20),
        (0.2, 1.0, 0.1, 20),
        (0.2, 1.0, -0.1, 20),
    ]

    for v, w, u, n_cells in parameter_sets:
        H = build_ssh_hamiltonian(v, w, u, n_cells)

        eigvals, pA_pos, pB_pos, pA_neg, pB_neg = solve_ssh_system(H)

        plot_probability_distribution(
            pA_pos,
            pB_pos,
            pA_neg,
            pB_neg,
            v,
            w,
            u,
        )

        plot_eigenvalue_spectrum(
            eigvals,
            v,
            w,
            u,
        )


if __name__ == "__main__":
    main()
