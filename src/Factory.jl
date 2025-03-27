function build(modeltype::Type{MyEpsilonGreedyAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;

    # return -
    return model;
end

function build(modeltype::Type{MyExploreFirstAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;

    # return -
    return model;
end

function build(modeltype::Type{MyUCB1AlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms

    # build empty model -
    model = modeltype();
    model.K = K;

    # return -
    return model;
end

function build(modeltype::MyBanditConsumerContextModel, data::NamedTuple)

    # initialize -
    m = data.m; # number of categories of goods
    γ = data.γ; # consumer's preference for each category of goods
    σ = data.σ; # uncetainty of consumer's preference for each category of goods
    β = data.β; # consumer's preference for each category of goods
    C = data.C; # price of each good in each category
    λ = data.λ; # how budget sensitive the consumer is
    B = data.B; # consumer's budget

    # build empty model -
    model = modeltype();
    model.m = m;
    model.γ = γ;
    model.σ = σ;
    model.β = β;
    model.C = C;
    model.λ = λ;
    model.B = B;

    # return -
    return model;
end