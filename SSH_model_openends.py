import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from pathlib import Path

OUTPUT_DIR = Path("Report/Figures")
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

def build_ssh_hamiltonian(v: float, w: float, n_cells: int) -> np.ndarray:
    """Constructs the Hamiltonian matrix for the Su-Schrieffer-Heeger (SSH) model."""
    dim = 2 * n_cells
    H = np.zeros((dim, dim))

    for m in range(n_cells):
        # Intra-cell hopping (v)
        H[2 * m, 2 * m + 1] = v
        H[2 * m + 1, 2 * m] = v

        # Inter-cell hopping (w)
        if m < n_cells - 1:
            H[2 * m + 1, 2 * m + 2] = w
            H[2 * m + 2, 2 * m + 1] = w

    return H

def solve_ssh_system(
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes all eigenvalues and extracts the target state probabilities."""
    # Compute all eigenvalues and eigenvectors using SciPy
    eigvals, eigvecs = la.eigh(H)

    # Find the smallest positive eigenvalue
    positive_indices = np.where(eigvals > 1e-12)[0]

    if len(positive_indices) == 0:
        raise ValueError("No positive eigenvalues found.")

    idx = positive_indices[0]
    psi = eigvecs[:, idx]

    # Calculate probabilities
    prob = np.abs(psi) ** 2
    prob_A = prob[0::2]
    prob_B = prob[1::2]

    return eigvals, prob_A, prob_B

def plot_probability_distribution(
    prob_A: np.ndarray, prob_B: np.ndarray, v: float, w: float
) -> None:
    """Plots the probability distribution across unit cells for sublattices A and B."""
    n_cells = len(prob_A)
    cells = np.arange(n_cells)
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    # Plot bars
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

    # Labels and Styling
    ax.set_xlabel("Unit Cell Index")
    ax.set_ylabel(r"Probability Density $|\psi|^2$")
    ax.set_xticks(cells)

    # Add a subtle grid behind the bars
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    filename = OUTPUT_DIR / f"probability_v={v}_w={w}_N={n_cells}.pdf"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

    print(f"Saved: {filename}")

def plot_eigenvalue_spectrum(eigvals: np.ndarray, v: float, w: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    indices = np.arange(len(eigvals))

    # Plot all eigenvalues
    ax.scatter(
        indices,
        eigvals,
        color="#2ca02c",
        edgecolor="black",
        s=45,
    )
    # Highlight edge states only in the topological phase
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
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)

    ax.set_xlabel("State Index")
    ax.set_ylabel(r"Energy $E$")
    ax.grid(True, linestyle="--", alpha=0.5)

    n_cells = len(eigvals) // 2
    filename = OUTPUT_DIR / f"spectrum_v={v}_w={w}_N={n_cells}.pdf"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"Saved: {filename}")



if __name__ == "__main__":
    # Parameters
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
    
    # Pipeline
    for v, w, N in parameter_sets:

        H = build_ssh_hamiltonian(v, w, N)

        eigvals, pA, pB = solve_ssh_system(H)

        plot_probability_distribution(
            pA, pB, v=v, w=w
        )

        plot_eigenvalue_spectrum(
            eigvals, v=v, w=w
        )