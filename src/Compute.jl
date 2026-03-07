# ============================================================================================= #
# Compute.jl
# --------------------------------------------------------------------------------------------- #
# Local (corrected) implementations of sample and learn for MyRestrictedBoltzmannMachineModel.
# These override the package methods to fix two bugs in the CD training loop:
#   Bug 1: weights were mutated in-place during batch processing, so later batch samples
#          saw partially updated weights instead of a consistent model state.
#   Bug 2: per-sample gradient was applied at full η without dividing by batch size,
#          making the effective learning rate scale linearly with batch size.
# ============================================================================================= #


"""
    sample(model::MyRestrictedBoltzmannMachineModel,
        vₒ::Vector{Int}; T::Int = 100, β::Float64 = 1.0) -> Tuple{Array{Int,2}, Array{Int,2}}

Run block Gibbs sampling on a restricted Boltzmann machine starting from visible state `vₒ`.

At each time step the method alternates between sampling the hidden layer conditioned on the
visible layer and sampling the visible layer conditioned on the hidden layer (block Gibbs).
The conditional probabilities use the {-1, +1} spin convention:

``P(h_k = +1 \\mid \\mathbf{v}) = \\sigma\\!\\left(2\\beta\\left(\\sum_j W_{jk}\\,v_j + b_k\\right)\\right)``

``P(v_j = +1 \\mid \\mathbf{h}) = \\sigma\\!\\left(2\\beta\\left(\\sum_k W_{jk}\\,h_k + a_j\\right)\\right)``

where ``\\sigma(x) = 1/(1+e^{-x})`` is the logistic sigmoid.

### Arguments
- `model::MyRestrictedBoltzmannMachineModel`: RBM with weight matrix `W`, hidden bias `b`,
  and visible bias `a`.
- `vₒ::Vector{Int}`: initial visible state in {-1, +1}ⁿ.
- `T::Int = 100`: number of Gibbs steps to record. The returned trajectories have `T` columns.
  For CD-k training, use `T = k + 1` (the first column stores the data-driven sample).
- `β::Float64 = 1.0`: inverse temperature parameter.

### Returns
- `(V, H)::Tuple{Array{Int,2}, Array{Int,2}}`: visible trajectory `V` (nᵥ × T) and hidden
  trajectory `H` (nₕ × T). Column 1 contains the initial (data-driven) state; column `T`
  contains the final (model-driven) state.
"""
function sample(model::MyRestrictedBoltzmannMachineModel,
    vₒ::Vector{Int}; T::Int = 100, β::Float64 = 1.0)

    # initialize -
    W = model.W; # weight matrix (nᵥ × nₕ)
    b = model.b; # hidden bias vector (nₕ)
    a = model.a; # visible bias vector (nᵥ)
    number_of_visible_neurons = length(a);
    number_of_hidden_neurons = length(b);

    # allocate storage for trajectories -
    V = zeros(Int, number_of_visible_neurons, T);
    H = zeros(Int, number_of_hidden_neurons, T);
    v = copy(vₒ);
    h = zeros(Int, number_of_hidden_neurons);
    IN_h = zeros(Float64, number_of_hidden_neurons);
    IN_v = zeros(Float64, number_of_visible_neurons);

    # step 1: store the initial visible state, sample h | vₒ -
    V[:, 1] .= v;
    for k ∈ 1:number_of_hidden_neurons
        IN_h[k] = dot(W[:, k], v) + b[k]; # input for hidden node k
    end
    for k ∈ 1:number_of_hidden_neurons
        pₖ = 1.0 / (1.0 + exp(-2.0 * β * IN_h[k])); # P(hₖ = +1 | v)
        h[k] = rand(Bernoulli(pₖ)) == 1 ? 1 : -1;
    end
    H[:, 1] .= h;

    # step 2: alternate block Gibbs updates for t = 2, …, T -
    for t ∈ 2:T

        # sample v | h -
        for j ∈ 1:number_of_visible_neurons
            IN_v[j] = dot(W[j, :], h) + a[j]; # input for visible node j
        end
        for j ∈ 1:number_of_visible_neurons
            pⱼ = 1.0 / (1.0 + exp(-2.0 * β * IN_v[j])); # P(vⱼ = +1 | h)
            v[j] = rand(Bernoulli(pⱼ)) == 1 ? 1 : -1;
        end

        # sample h | v -
        for k ∈ 1:number_of_hidden_neurons
            IN_h[k] = dot(W[:, k], v) + b[k]; # input for hidden node k
        end
        for k ∈ 1:number_of_hidden_neurons
            pₖ = 1.0 / (1.0 + exp(-2.0 * β * IN_h[k])); # P(hₖ = +1 | v)
            h[k] = rand(Bernoulli(pₖ)) == 1 ? 1 : -1;
        end

        # store the current state -
        V[:, t] .= v;
        H[:, t] .= h;
    end

    # return -
    return (V, H);
end

"""
    learn(model::MyRestrictedBoltzmannMachineModel,
        data::Array{Int64,2}, p::Categorical;
        maxnumberofiterations::Int = 100, T::Int = 100, β::Float64 = 1.0,
        batchsize::Int = 10, η::Float64 = 0.01, tol::Float64 = 1e-6,
        verbose::Bool = true) -> MyRestrictedBoltzmannMachineModel

Train a restricted Boltzmann machine using the contrastive divergence (CD) algorithm.

Each iteration samples a mini-batch of `batchsize` unique training examples, runs `T`-step
block Gibbs sampling on each example to estimate the negative phase, then applies a single
batch-averaged gradient step. With `T = k + 1`, this implements CD-k.

The gradient for a single training example ``\\mathbf{x}`` is:

``\\Delta W_{jk} = x_j\\,h_k^{(1)} - v_j^{(T)}\\,h_k^{(T)}``

``\\Delta b_k = h_k^{(1)} - h_k^{(T)}``

``\\Delta a_j = x_j - v_j^{(T)}``

where ``(\\cdot)^{(1)}`` is the data-driven (positive) phase and ``(\\cdot)^{(T)}`` is the
model-driven (negative) phase. The batch-averaged update is:

``W \\leftarrow W + \\frac{\\eta}{B}\\sum_{i=1}^{B} \\Delta W^{(i)}``

### Arguments
- `model::MyRestrictedBoltzmannMachineModel`: RBM to train.
- `data::Array{Int64,2}`: training data matrix (nᵥ × N), each column a pattern in {-1, +1}ⁿ.
- `p::Categorical`: sampling distribution over column indices of `data`.
- `maxnumberofiterations::Int = 100`: maximum number of weight update iterations.
- `T::Int = 100`: Gibbs sampling steps per example. Use `T = k + 1` for CD-k.
- `β::Float64 = 1.0`: inverse temperature for sampling.
- `batchsize::Int = 10`: number of unique examples per mini-batch.
- `η::Float64 = 0.01`: learning rate applied to the batch-averaged gradient.
- `tol::Float64 = 1e-6`: early stopping tolerance on relative parameter change.
- `verbose::Bool = true`: print iteration progress if true.

### Returns
- `MyRestrictedBoltzmannMachineModel`: a new model with the trained parameters.
"""
function learn(model::MyRestrictedBoltzmannMachineModel,
    data::Array{Int64,2}, p::Categorical;
    maxnumberofiterations::Int = 100, T::Int = 100, β::Float64 = 1.0,
    batchsize::Int = 10, η::Float64 = 0.01,
    tol::Float64 = 1e-6, verbose::Bool = true)::MyRestrictedBoltzmannMachineModel

    # initialize - copy parameters so we do not mutate the input model
    W = copy(model.W); # weight matrix (nᵥ × nₕ)
    b = copy(model.b); # hidden bias vector (nₕ)
    a = copy(model.a); # visible bias vector (nᵥ)
    number_of_visible_neurons = size(W, 1);
    number_of_hidden_neurons = size(W, 2);
    counter = 1;
    is_ok_to_stop = false;

    # main training loop -
    while (is_ok_to_stop == false)

        # stash current parameters for convergence check -
        W_prev = copy(W);
        b_prev = copy(b);
        a_prev = copy(a);

        # build a frozen snapshot of the model for this iteration's sampling.
        # all batch samples must see the SAME weights (this was bug 1) -
        frozen_model = build(MyRestrictedBoltzmannMachineModel, (
            W = copy(W), # frozen weight matrix
            b = copy(b), # frozen hidden bias
            a = copy(a)  # frozen visible bias
        ));

        # initialize gradient accumulators (zeroed each iteration) -
        ΔW = zeros(Float64, number_of_visible_neurons, number_of_hidden_neurons);
        Δb = zeros(Float64, number_of_hidden_neurons);
        Δa = zeros(Float64, number_of_visible_neurons);

        # sample a mini-batch of unique training indices -
        idx_batch_set = Set{Int64}();
        while (length(idx_batch_set) < batchsize)
            push!(idx_batch_set, rand(p));
        end
        idx_batch = idx_batch_set |> collect |> sort;

        # accumulate gradients over the mini-batch -
        for i ∈ eachindex(idx_batch)
            idx = idx_batch[i]; # training example index
            xₒ = data[:, idx];  # data vector (positive phase visible)

            # run block Gibbs sampling from xₒ using the FROZEN model -
            (V_chain, H_chain) = sample(frozen_model, xₒ, T = T, β = β);

            # extract positive and negative phase statistics -
            h_pos = H_chain[:, 1];   # h sampled from data (positive phase)
            v_neg = V_chain[:, end]; # v after T-1 Gibbs steps (negative phase)
            h_neg = H_chain[:, end]; # h after T-1 Gibbs steps (negative phase)

            # accumulate: ⟨v hᵀ⟩_data − ⟨v hᵀ⟩_model -
            ΔW .+= xₒ * h_pos' .- v_neg * h_neg';
            Δb .+= h_pos .- h_neg;
            Δa .+= xₒ .- v_neg;
        end

        # apply the batch-averaged gradient update (this was bug 2:
        # the old code applied η per sample instead of η/batchsize) -
        scale = η / batchsize;
        W .+= scale .* ΔW;
        b .+= scale .* Δb;
        a .+= scale .* Δa;

        # compute relative parameter change for early stopping -
        ΔW_rel = norm(W .- W_prev) / (norm(W_prev) + eps(Float64));
        Δb_rel = norm(b .- b_prev) / (norm(b_prev) + eps(Float64));
        Δa_rel = norm(a .- a_prev) / (norm(a_prev) + eps(Float64));
        parameter_change = max(ΔW_rel, Δb_rel, Δa_rel);

        if (verbose == true)
            println("Iteration: ", counter, ", max relative parameter change: ", parameter_change);
        end

        # check stopping criteria -
        if (counter ≥ maxnumberofiterations || parameter_change < tol)
            is_ok_to_stop = true;
        else
            counter += 1;
        end
    end

    # build and return a new model with the trained parameters -
    trained_model = build(MyRestrictedBoltzmannMachineModel, (
        W = W, # trained weight matrix
        b = b, # trained hidden bias
        a = a  # trained visible bias
    ));

    # return -
    return trained_model;
end
