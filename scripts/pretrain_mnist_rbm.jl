"""
pretrain_mnist_rbm.jl
----------------------
Trains a Restricted Boltzmann Machine on MNIST handwritten digits (all 10 classes)
and saves the result to data/pretrained_rbm_mnist.jld2.

Run from the project root:
    julia scripts/pretrain_mnist_rbm.jl

Outputs:
    data/pretrained_rbm_mnist.jld2  - trained RBM weights and biases

Training algorithm: Contrastive Divergence (CD-4)
    Each epoch runs ceil(N_total / BATCH_SIZE) mini-batch weight updates.
    T=5 means four Gibbs steps (initial state + four updates = CD-4).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# load everything (packages + corrected local methods) via Include.jl -
include(joinpath(@__DIR__, "..", "Include.jl"))

Random.seed!(2025)

# ─── configuration ────────────────────────────────────────────────────────────

const N_EXAMPLES_PER_DIGIT = 200   # MNIST examples per digit class (0-9)
const N_VISIBLE  = 784             # 28 x 28 pixels
const N_HIDDEN   = 512             # hidden units
const N_EPOCHS   = 300             # training epochs
const BATCH_SIZE = 50              # mini-batch size
const ETA        = 0.1             # learning rate
const BETA       = 1.0             # inverse temperature
const T_GIBBS    = 5               # Gibbs steps (T=5 -> CD-4)
const TOL        = 1e-10           # convergence tolerance
const LOG_EVERY  = 10              # print progress every N epochs

# ─── load MNIST ───────────────────────────────────────────────────────────────

println("Loading MNIST ($N_EXAMPLES_PER_DIGIT examples per digit)...")
digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = N_EXAMPLES_PER_DIGIT)

N_total = 10 * N_EXAMPLES_PER_DIGIT
X_all   = zeros(Int64, N_VISIBLE, N_total)

for (digit_idx, digit) in enumerate(0:9)
    for i in 1:N_EXAMPLES_PER_DIGIT
        col           = (digit_idx - 1) * N_EXAMPLES_PER_DIGIT + i
        img           = digits_dict[digit][:, :, i]
        v             = Float64.(img)[:]
        b             = (v .> 0.5)
        X_all[:, col] = Int64.(2 .* b .- 1)
    end
end

println("Training data: $(N_VISIBLE) pixels x $(N_total) examples")

p_data = Categorical(N_total)

# ─── build RBM ───────────────────────────────────────────────────────────────

println("\nInitializing RBM: $(N_VISIBLE) visible -> $(N_HIDDEN) hidden")

rbm = build(MyRestrictedBoltzmannMachineModel, (
    W = 0.01 * randn(N_VISIBLE, N_HIDDEN),
    b = zeros(N_HIDDEN),
    a = zeros(N_VISIBLE)
))

# ─── training loop ────────────────────────────────────────────────────────────

n_updates = ceil(Int, N_total / BATCH_SIZE)
println("Training (CD-4, eta=$(ETA), beta=$(BETA), batch=$(BATCH_SIZE), epochs=$(N_EPOCHS))...\n")

for epoch in 1:N_EPOCHS
    global rbm = learn(rbm, X_all, p_data;
        maxnumberofiterations = n_updates,
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

# ─── save ─────────────────────────────────────────────────────────────────────

data_dir = joinpath(@__DIR__, "..", "data")
out_path = joinpath(data_dir, "pretrained_rbm_mnist.jld2")

jldsave(out_path;
    W          = rbm.W,
    b          = rbm.b,
    a          = rbm.a,
    n_visible  = N_VISIBLE,
    n_hidden   = N_HIDDEN,
    beta       = BETA,
    n_examples = N_EXAMPLES_PER_DIGIT)

println("Saved -> $(out_path)  ($(N_VISIBLE) -> $(N_HIDDEN), beta=$(BETA))")
