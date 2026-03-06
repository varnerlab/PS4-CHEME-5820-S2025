"""
train_pretrained_rbm.jl
-----------------------
Trains a Restricted Boltzmann Machine on the FDA drug fingerprint dataset
and saves the result to data/pretrained_rbm_drugs.jld2.

Run from the project root:
    julia scripts/train_pretrained_rbm.jl

Inputs:
    data/fda_drugs_fingerprints.jld2   - fingerprints (256 × N Float32) + drug names

Outputs:
    data/pretrained_rbm_drugs.jld2     - trained RBM weights and biases

Training algorithm: Contrastive Divergence (CD-k)
    CD-k approximates the log-likelihood gradient by running k steps of block
    Gibbs sampling from the data state rather than to equilibrium. In practice
    CD-1 (T=2 in this package, which stores the initial state + one Gibbs step)
    converges well and is much faster than running chains to equilibrium.

    Each iteration updates weights and biases as:
        ΔW  ∝  η · (⟨v h⟩_data  −  ⟨v h⟩_model)
        Δa  ∝  η · (⟨v⟩_data    −  ⟨v⟩_model)
        Δb  ∝  η · (⟨h⟩_data    −  ⟨h⟩_model)

    where ⟨·⟩_data is the expectation under the data distribution and
    ⟨·⟩_model is approximated by the CD-k chain endpoint.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Random, Statistics, LinearAlgebra
using Distributions: Categorical
using VLDataScienceMachineLearningPackage
import StatsBase

Random.seed!(2025)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — change any of these values to experiment
# ──────────────────────────────────────────────────────────────────────────────

# Model architecture
const N_VISIBLE = 256;   # number of visible units (must match fingerprint dimension)
const N_HIDDEN  = 512;   # number of hidden units

# Training hyperparameters
const N_EPOCHS   = 200;   # total number of training epochs (each = one full pass over data)
const BATCH_SIZE = 32;    # number of samples per mini-batch
const ETA        = 0.01;  # learning rate η
const BETA       = 1.0;   # inverse temperature
const T_GIBBS    = 2;     # number of Gibbs steps T (T=2 → CD-1)
const TOL        = 1e-10  # convergence tolerance (set high to rely on N_EPOCHS)

# Logging
const LOG_EVERY  = 10;   # print a progress message every this many epochs

# ──────────────────────────────────────────────────────────────────────────────
# Load fingerprints
# ──────────────────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "..", "data")

fp_file      = jldopen(joinpath(data_dir, "fda_drugs_fingerprints.jld2"))
fingerprints = fp_file["fingerprints"]   # Float32, N_VISIBLE × N_total
drug_names   = fp_file["drug_names"]     # Vector{String}
close(fp_file)

N_total = size(fingerprints, 2)
println("Loaded fingerprints: $(size(fingerprints, 1)) bits × $(N_total) drugs")

# Convert {0,1} Float32  →  {-1,+1} Int64  (required by the RBM)
fp_pm1 = Int64.(2 .* fingerprints .- 1)

# Uniform categorical distribution over all training examples
p_data = Categorical(N_total)

# ──────────────────────────────────────────────────────────────────────────────
# Build RBM
# ──────────────────────────────────────────────────────────────────────────────
println("\nInitializing RBM: $(N_VISIBLE) visible → $(N_HIDDEN) hidden")

rbm = build(MyRestrictedBoltzmannMachineModel, (
    W = 0.01 * randn(N_VISIBLE, N_HIDDEN),  # small random weights
    b = zeros(N_HIDDEN),                     # hidden biases (start at zero)
    a = zeros(N_VISIBLE),                    # visible biases (start at zero)
))

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
println("Training (CD-$(T_GIBBS-1), η=$(ETA), β=$(BETA), batch=$(BATCH_SIZE), epochs=$(N_EPOCHS))...\n")

n_updates_per_epoch = ceil(Int, N_total / BATCH_SIZE)  # one full pass over data

for epoch in 1:N_EPOCHS
    global rbm = learn(rbm, fp_pm1, p_data;
        maxnumberofiterations = n_updates_per_epoch,
        T         = T_GIBBS,
        β         = BETA,
        batchsize = BATCH_SIZE,
        η         = ETA,
        tol       = TOL,
        verbose   = false)

    if epoch % LOG_EVERY == 0
        println("  Epoch $(lpad(epoch, 4)) / $(N_EPOCHS) complete")
    end
end

println("\nTraining complete.")

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
out_path = joinpath(data_dir, "pretrained_rbm_drugs.jld2")

jldsave(out_path;
    W         = rbm.W,
    b         = rbm.b,
    a         = rbm.a,
    n_visible = N_VISIBLE,
    n_hidden  = N_HIDDEN,
    beta      = BETA)

println("Saved → $(out_path)  ($(N_VISIBLE) → $(N_HIDDEN), β=$(BETA))")
