# PS4: Restricted Boltzmann Machines and MNIST Digit Recall

## Overview
This problem set explores how two associative memory models — the classical Hopfield network and the restricted Boltzmann machine (RBM) — store and recall handwritten digit patterns from corrupted inputs using the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

## Tasks
- **Task 1:** Build a Hopfield network memory matrix from MNIST digit patterns and test recall from corrupted inputs using Hebbian dynamics.
- **Task 2:** Train a small RBM (784 visible, 128 hidden) on a single digit class using contrastive divergence (CD-1) and evaluate recall quality.
- **Task 3:** Load a pretrained RBM (784 visible, 512 hidden) trained on all ten digit classes and compare recall performance against the small RBM across varying corruption levels.

## Discussion Questions
Three discussion questions ask students to explore Hopfield storage capacity limits, the effect of Gibbs sampling steps on RBM recall, and the specialization-generalization tradeoff between single-class and multi-class models.

## Repository Structure
```
├── Include.jl                          # Environment setup (packages + local overrides)
├── src/
│   └── Compute.jl                      # Corrected sample() and learn() implementations
├── scripts/
│   └── pretrain_mnist_rbm.jl           # Pretraining script for the large RBM
├── data/
│   └── pretrained_rbm_mnist.jld2       # Pretrained RBM weights (784 -> 512)
├── PS4-CHEME-5820-Solution-RBM-Digits-S2025.ipynb  # Solution notebook
└── PS4-CHEME-5820-Student-RBM-Digits-S2025.ipynb   # Student notebook (with TODO placeholders)
```

## Getting Started
1. Open the student notebook `PS4-CHEME-5820-Student-RBM-Digits-S2025.ipynb` in VS Code or Jupyter.
2. Run the first cell to load the environment via `Include.jl`.
3. Complete the `TODO` sections in each task, then answer the discussion questions.
4. Run the test cell at the bottom to verify your results.

## Pretraining (optional)
The pretrained RBM is provided in `data/`. To retrain from scratch:
```bash
julia --project=. scripts/pretrain_mnist_rbm.jl
```
This takes several hours depending on hardware.

## Dependencies
All dependencies are managed via `Project.toml`. The primary package is [`VLDataScienceMachineLearningPackage.jl`](https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl). Local corrected implementations of `sample` and `learn` in `src/Compute.jl` override the package versions.
