import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from pathlib import Path

OUTPUT_DIR = Path("figures") / "onsite_potential"
OUTPUT_DIR.mkdir(exist_ok=True)

# Plotting parameters
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
    """Constructs the Hamiltonian matrix for the Su-Schrieffer-Heeger (SSH) model."""
    dim = 2 * n_cells
    H = np.zeros((dim, dim))

    for m in range(n_cells):
        A = 2 * m
        B = 2 * m + 1
        # Intra-cell hopping (v)
        H[A, B] = v
        H[B, A] = v

        # Inter-cell hopping (w)
        if m < n_cells - 1:
            H[B, A + 2] = w
            H[A + 2, B] = w
        
        H[A, A] = u  # On-site potential for sublattice A
        H[B, B] = -u  # On-site potential for sublattice B

    return H

def solve_ssh_system(
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes all eigenvalues and extracts the target state probabilities."""
    # Compute all eigenvalues and eigenvectors using SciPy
    eigvals, eigvecs = la.eigh(H)

    # Smallest positive-energy edge state
    positive_idx = np.where(eigvals > 1e-30)[0][0]

    # Largest negative-energy edge state
    negative_idx = np.where(eigvals < -1e-30)[0][-1]

    psi_pos = eigvecs[:, positive_idx]
    psi_neg = eigvecs[:, negative_idx]

    prob_pos = np.abs(psi_pos)**2
    prob_neg = np.abs(psi_neg)**2

    prob_A_pos = prob_pos[0::2]
    prob_B_pos = prob_pos[1::2]

    prob_A_neg = prob_neg[0::2]
    prob_B_neg = prob_neg[1::2]

    return (
        eigvals,
        prob_A_pos,
        prob_B_pos,
        prob_A_neg,
        prob_B_neg,
        )

def plot_probability_distribution(
    prob_A_pos: np.ndarray,
    prob_B_pos: np.ndarray,
    prob_A_neg: np.ndarray,
    prob_B_neg: np.ndarray,
    v: float,
    w: float,
    u: float,
) -> None:
    """Plots the positive- and negative-energy edge-state probability distributions."""

    n_cells = len(prob_A_pos)
    cells = np.arange(n_cells)
    width = 0.4

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        dpi=100,
        sharey=True
    )

    # ---------------- Positive-energy edge state ----------------
    ax = axes[0]

    ax.bar(
        cells - width/2,
        prob_A_pos,
        width,
        label="Sublattice A",
        color="#1f77b4",
        edgecolor="black",
    )

    ax.bar(
        cells + width/2,
        prob_B_pos,
        width,
        label="Sublattice B",
        color="#ff7f0e",
        edgecolor="black",
    )

    ax.set_title("Positive Edge State")
    ax.set_xlabel("Unit Cell Index")
    ax.set_ylabel(r"Probability Density $|\psi|^2$")
    ax.set_xticks(cells)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    # ---------------- Negative-energy edge state ----------------
    ax = axes[1]

    ax.bar(
        cells - width/2,
        prob_A_neg,
        width,
        label="Sublattice A",
        color="#1f77b4",
        edgecolor="black",
    )

    ax.bar(
        cells + width/2,
        prob_B_neg,
        width,
        label="Sublattice B",
        color="#ff7f0e",
        edgecolor="black",
    )

    ax.set_title("Negative Edge State")
    ax.set_xlabel("Unit Cell Index")
    ax.set_xticks(cells)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=True)


    plt.tight_layout()

    filename = OUTPUT_DIR / (
        f"edge_states_v={v}_w={w}_u={u}_N={n_cells}.pdf"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    print(f"Saved: {filename}")

def plot_eigenvalue_spectrum(eigvals: np.ndarray, v: float, w: float, u: float) -> None:
    """Plots the complete eigenvalue energy spectrum of the SSH system."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    positive_idx = np.where(eigvals > 1e-30)[0][0]
    negative_idx = np.where(eigvals < -1e-30)[0][-1]
    
    indices = np.arange(len(eigvals))

    # Bulk eigenvalues
    ax.scatter(
        indices,
        eigvals,
        color="#2ca02c",
        edgecolor="black",
        s=45,
        zorder=2,
    )

    # Highlight the two edge states
    ax.scatter(
        negative_idx,
        eigvals[negative_idx],
        color="red",
        edgecolor="black",
        s=90,
        zorder=4,
    )

    ax.scatter(
        positive_idx,
        eigvals[positive_idx],
        color="red",
        edgecolor="black",
        s=90,
        zorder=4,
    )

    # Reference line at zero-energy
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)

    # Labels and Styling
    ax.set_xlabel("State Index")
    ax.set_ylabel("Energy $E$")
    n_cells = len(eigvals)//2
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    filename = OUTPUT_DIR / f"spectrum_v={v}_w={w}_u={u}_N={n_cells}.pdf"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {filename}")



if __name__ == "__main__":
    # Parameters (v, w, u, N) where u is not used in the current implementation but can be included for future extensions
    parameter_sets = [
    (0.2, 1.0, 0.0, 10),
    (0.2, 1.0, 0.1, 10),
    (0.2, 1.0, -0.1, 10),
    
    (0.2, 1.0, 0.0, 20),
    (0.2, 1.0, 0.1, 20),
    (0.2, 1.0, -0.1, 20),   
]
    # Pipeline
    for v, w, u, N in parameter_sets:

        H = build_ssh_hamiltonian(v, w, u, N)

        eigvals, pA_pos, pB_pos, pA_neg, pB_neg = solve_ssh_system(H)


        plot_probability_distribution(
            pA_pos,
            pB_pos,
            pA_neg,
            pB_neg,
            v=v,
            w=w,
            u=u,
        )

        plot_eigenvalue_spectrum(
            eigvals, v=v, w=w, u=u
        )