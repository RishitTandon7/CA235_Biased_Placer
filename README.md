# 🛣️ DREAMPlace MetaOpt

## Metaheuristic VLSI Placement — No Deep Learning

This project is a **reimplementation of [DREAMPlace](https://github.com/limbo018/DREAMPlace)** that **removes the deep learning / PyTorch dependency** and replaces the gradient-based optimizer with three metaheuristic algorithms.

---

## 🚀 Quick Start

**Double-click `run.bat`** or run:

```bash
python run.py
```

This will:
1. Auto-install any missing Python packages
2. Generate a synthetic BigBlue4-like benchmark
3. Open the **4-panel GUI** where you can select and run algorithms

---

## ⚙️ Algorithms

| # | Algorithm | Description |
|---|-----------|-------------|
| 1 | **Hybrid SBO** | Surrogate-Based Optimization with RBF interpolation + Nelder-Mead local refinement |
| 2 | **PSO + SBO** | Particle Swarm Optimization with surrogate-assisted fitness evaluation |
| 3 | **CA235** | Cellular Automata Rule 235 — density-driven cell spreading + wirelength refinement |

All three replace the PyTorch Adam/Nesterov optimizer used in the original DREAMPlace.

---

## 🖥️ GUI Output Panels

| Panel | Shows |
|-------|-------|
| **Density Map** | Heatmap of cell density in each grid bin |
| **Electric Potential** | Potential φ from Poisson equation (FFT/DCT) |
| **Electric Field** | Field E = -∇φ with vector arrows |
| **Cell Grid** | Cells placed on discretized grid (green=movable, orange=fixed) |

---

## 📂 Project Structure

```
DREAMPlace_MetaOpt/
├── run.py                      # ← Single-click entry point
├── run.bat                     # ← Windows launcher
├── requirements.txt
├── core/
│   ├── placement.py            # Data structures (NumPy-based)
│   ├── benchmark_parser.py     # ISPD parser + synthetic generator
│   ├── density.py              # Bin-level density computation
│   ├── potential.py            # Poisson solver (DCT/FFT)
│   ├── field.py                # Electric field = -∇φ
│   ├── wirelength.py           # HPWL computation
│   └── objectives.py           # Combined objective function
├── algorithms/
│   ├── hybrid_sbo.py           # Hybrid SBO
│   ├── pso_sbo.py              # PSO + SBO
│   └── cellular_automata.py    # Cellular Automata CA235
├── gui/
│   └── visualizer.py           # 4-panel Tkinter/Matplotlib GUI
├── results/                    # Saved output plots
└── benchmarks/                 # Place ISPD .aux files here
```

---

## 🔧 CLI Options

```bash
python run.py                           # GUI mode (default)
python run.py --algo sbo                # Run only Hybrid SBO
python run.py --algo pso                # Run only PSO+SBO
python run.py --algo ca235              # Run only CA235
python run.py --cells 10000             # Use 10K synthetic cells
python run.py --benchmark path/to.aux   # Load real ISPD benchmark
python run.py --no-gui                  # Console only (saves plots to results/)
```

---

## 📋 Requirements

- **Python 3.8+** (no GPU needed!)
- Packages: `numpy`, `scipy`, `matplotlib`, `scikit-learn`
- All packages are **auto-installed** on first run

---

## 🏗️ How It Works

### Original DREAMPlace (Deep Learning)
- Cell positions = "trainable weights" in a neural network
- HPWL + density penalty = "loss function"
- Optimization via PyTorch autograd + Adam/Nesterov

### This Project (Metaheuristic)
- Cell positions = decision variables
- Same HPWL + density penalty objective
- **Surrogate models (RBF)** approximate the objective landscape
- **PSO particles** explore the search space
- **Cellular Automata** handle density spreading
- **No gradients, no backprop, no GPU required**

---

## 📊 BigBlue4 Benchmark

The BigBlue4 benchmark (ISPD 2005) has 2.1M cells. For tractability:
- The synthetic generator creates a **scaled-down** version with configurable cell count
- Algorithms use **cell clustering** (k-means) to reduce dimensionality
- If you have the real BigBlue4 files, place the `.aux` file in `benchmarks/` and use `--benchmark`
