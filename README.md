# Summer Research Internship 2026
## A Theoretical and Numerical Study of the Su–Schrieffer–Heeger (SSH) Model

This repository contains the work completed during my Summer 2026 research internship under the supervision of **Dr. Panch Ram** (Department of Physics, IIT (BHU), Varanasi).

The project investigates the **Su–Schrieffer–Heeger (SSH) model**, one of the simplest lattice models exhibiting topological phases of matter. Starting from the real-space tight-binding Hamiltonian, the model is developed analytically and explored numerically through Python simulations to study its topological properties, edge states, and the effects of chiral-symmetry breaking.

---

## Objectives

The primary objectives of this internship were to:

- Develop the SSH model from first principles
- Derive the bulk momentum-space Hamiltonian
- Obtain the dispersion relation
- Study chiral symmetry and its consequences
- Understand the winding number as a topological invariant
- Investigate bulk–boundary correspondence
- Numerically diagonalize finite SSH chains
- Study edge-state localization
- Examine the effect of a staggered on-site potential on the system's topology

---

## Repository Structure

```
ssh-model-study/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── ssh_model.py
│   ├── ssh_open_boundary.py
│   ├── ssh_onsite.py
│   └── winding_number.py
│
├── Reading/
│
└── Report/
    ├── Figures/
    ├── main.tex
    ├── preamble.tex
    ├── References.bib
    └── SSH_Model_Internship_Report.pdf
```

---

## Python Simulations

The repository contains four Python programs, located in `code/`.

### `ssh_model.py`

Studies the bulk SSH model with periodic boundary conditions.

Features:
- Energy dispersion relation
- Winding trajectories in the $d_x$ – $d_y$ plane
- Visualization of the topological phase transition
- Publication-quality figures using Matplotlib

### `ssh_open_boundary.py`

Numerically diagonalizes finite SSH chains with open boundary conditions.

Features:
- Construction of finite Hamiltonians
- Exact numerical diagonalization
- Complete energy spectrum
- Edge-state probability distributions
- Comparison of trivial and topological phases

### `ssh_onsite.py`

Extends the SSH model by introducing a staggered on-site potential.

Features:
- Modified Hamiltonian with on-site energies
- Evolution of edge states
- Energy spectrum
- Localization of positive- and negative-energy edge states
- Investigation of chiral-symmetry breaking

### `winding_number.py`

Visualizes the geometric interpretation of topology after adding an on-site potential.

Features:
- Three-dimensional $d$-vector trajectories
- Projection onto the $d_x$ – $d_y$ plane
- Illustration of the loss of the planar winding number

---

## Topics Covered

- Tight-binding approximation
- Bloch's theorem
- Periodic and open boundary conditions
- Band structure
- Topological phase transition
- Chiral symmetry
- Winding number
- Bulk–boundary correspondence
- Edge states
- Symmetry breaking
- Numerical diagonalization

---

## Software Requirements

- Python 3.13+
- NumPy
- SciPy
- Matplotlib
- A LaTeX distribution (for compiling the report, with `latexmk` recommended)

Install the Python dependencies with:

```bash
pip install numpy scipy matplotlib
```

> Note: the plotting scripts set `text.usetex: True` for publication-quality typesetting, which requires a working LaTeX installation on your system (e.g. TeX Live or MiKTeX).

---

## Report

The complete internship report is included as:

```
Report/main.pdf
Report/SSH_Model_Internship_Report.pdf
```

The report contains:
- Analytical derivations (real-space and momentum-space Hamiltonians, chiral symmetry, winding number, bulk–boundary correspondence)
- Numerical methods and Python implementations
- Simulation results (dispersion relations, winding trajectories, energy spectra, edge-state localization)
- Extension of the SSH model with a staggered on-site potential
- Full appendix reproducing all Python source code

To rebuild the report from source:

```bash
cd Report
latexmk -pdf main.tex
```

---

## Selected References

- W. P. Su, J. R. Schrieffer, and A. J. Heeger, *Solitons in Polyacetylene*, Physical Review Letters **42**, 1698 (1979)
- J. K. Asbóth, L. Oroszlány, and A. Pályi, *A Short Course on Topological Insulators* (Springer, 2015)
- M. Z. Hasan and C. L. Kane, *Colloquium: Topological Insulators*, Reviews of Modern Physics **82**, 3045 (2010)
- X.-L. Qi and S.-C. Zhang, *Topological Insulators and Superconductors*, Reviews of Modern Physics **83**, 1057 (2011)
- N. W. Ashcroft and N. D. Mermin, *Solid State Physics* (Holt, Rinehart and Winston, 1976)
- David J. Griffiths, *Introduction to Quantum Mechanics*, 3rd ed.

Full reading material is archived in `Reading/`.

---

## Acknowledgements

I sincerely thank **Dr. Panch Ram** (Department of Physics, IIT (BHU), Varanasi) for his guidance, insightful discussions, and supervision throughout this internship.

---

## Author

**Vijay Vedanth Vemula**
BS–MS Student
Indian Institute of Science Education and Research (IISER) Tirupati
Summer Research Internship 2026
