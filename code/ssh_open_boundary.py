"""
Finite SSH Chain with Open Boundary Conditions
==============================================

Author
------
Vijay Vedanth Vemula

Description
-----------
This script constructs the finite Su-Schrieffer-Heeger (SSH) Hamiltonian
with open boundary conditions and numerically diagonalizes it to study
the energy spectrum and topological edge states.

For a given set of hopping amplitudes, the script:

- Constructs the finite SSH Hamiltonian.
- Computes all eigenvalues and eigenvectors.
- Identifies the lowest positive-energy edge state.
- Plots the edge-state probability distribution.
- Plots the complete energy spectrum.
- Saves all generated figures in Report/Figures/.

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

OUTPUT_DIR = Path("Report/Figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Plotting parameters
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


def build_ssh_hamiltonian(v: float, w: float, n_cells: int) -> np.ndarray:
    """
    Construct the finite SSH Hamiltonian with open boundaries.

    Parameters
    ----------
    v : float
        Intracell hopping amplitude.
    w : float
        Intercell hopping amplitude.
    n_cells : int
        Number of unit cells.

    Returns
    -------
    numpy.ndarray
        SSH Hamiltonian of dimension (2N × 2N).
    """
    dim = 2 * n_cells
    H = np.zeros((dim, dim))

    for m in range(n_cells):
        A = 2 * m
        B = A + 1

        # Intracell hopping
        H[A, B] = v
        H[B, A] = v

        # Intercell hopping
        if m < n_cells - 1:
            H[B, A + 2] = w
            H[A + 2, B] = w

    return H


def solve_ssh_system(H: np.ndarray):
    """
    Diagonalize the SSH Hamiltonian and extract the lowest positive-energy
    edge state.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues.
    prob_A : ndarray
        Probability density on sublattice A.
    prob_B : ndarray
        Probability density on sublattice B.
    """
    eigvals, eigvecs = la.eigh(H)

    positive_indices = np.where(eigvals > 1e-12)[0]
    if len(positive_indices) == 0:
        raise ValueError("No positive eigenvalues found.")

    psi = eigvecs[:, positive_indices[0]]

    probability = np.abs(psi) ** 2

    return (
        eigvals,
        probability[0::2],
        probability[1::2],
    )


def plot_probability_distribution(
    prob_A: np.ndarray,
    prob_B: np.ndarray,
    v: float,
    w: float,
) -> None:
    """Plot the edge-state probability distribution."""

    n_cells = len(prob_A)
    cells = np.arange(n_cells)
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    ax.bar(
        cells - width / 2,
        prob_A,
        width,
        label="Sublattice A",
        color="#1f77b4",
        linewidth=0,
        alpha=0.8,
    )

    ax.bar(
        cells + width / 2,
        prob_B,
        width,
        label="Sublattice B",
        color="#ff7f0e",
        linewidth=0,
        alpha=0.8,
    )

    ax.set_xlabel("Unit Cell Index")
    ax.set_ylabel(r"Probability Density $|\psi|^2$")
    ax.set_xticks(cells)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    filename = OUTPUT_DIR / f"probability_v={v}_w={w}_N={n_cells}.pdf"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


def plot_eigenvalue_spectrum(
    eigvals: np.ndarray,
    v: float,
    w: float,
) -> None:
    """Plot the complete eigenvalue spectrum."""

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    indices = np.arange(len(eigvals))

    ax.scatter(
        indices,
        eigvals,
        color="#2ca02c",
        edgecolor="black",
        s=45,
    )

    if w > v:
        positive_idx = np.where(eigvals > 1e-36)[0][0]
        negative_idx = np.where(eigvals < -1e-36)[0][-1]

        ax.scatter(
            [negative_idx, positive_idx],
            [eigvals[negative_idx], eigvals[positive_idx]],
            color="red",
            edgecolor="black",
            s=90,
            label="Edge states",
            zorder=5,
        )

        ax.legend()

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel("State Index")
    ax.set_ylabel(r"Energy $E$")
    ax.grid(True, linestyle="--", alpha=0.5)

    filename = OUTPUT_DIR / (
        f"spectrum_v={v}_w={w}_N={len(eigvals)//2}.pdf"
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


def main():
    """Run all SSH simulations."""

    parameter_sets = [
        (0.2, 1.0, 10),
        (0.5, 1.0, 10),
        (1.0, 1.0, 10),
        (1.5, 1.0, 10),
        (0.2, 1.0, 20),
        (0.5, 1.0, 20),
        (1.0, 1.0, 20),
        (1.5, 1.0, 20),
    ]

    for v, w, n_cells in parameter_sets:

        H = build_ssh_hamiltonian(v, w, n_cells)

        eigvals, prob_A, prob_B = solve_ssh_system(H)

        plot_probability_distribution(
            prob_A,
            prob_B,
            v,
            w,
        )

        plot_eigenvalue_spectrum(
            eigvals,
            v,
            w,
        )


if __name__ == "__main__":
    main()
