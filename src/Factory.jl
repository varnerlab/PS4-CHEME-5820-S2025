function build(modeltype::Type{MyEpsilonGreedyAlgorithmModel}, data::NamedTuple)

    # initialize -
    K = data.K; # number of arms
    n = data.n; # recomended quantity of each good in each category

    # build empty model -
    model = modeltype();
    model.K = K;
    model.n = n;

    # return -
    return model;
end


function build(modeltype::Type{MyBanditConsumerContextModel}, data::NamedTuple)::MyBanditConsumerContextModel

    # initialize -
    m = data.m; # number of arms categories of goods
    γ = data.γ; # consumer's preference for each category of goods
    σ = data.σ; # uncetainty of consumer's preference for each category of goods
    Z = data.Z; # consumer's error model
    C = data.C; # price of each good in each category
    λ = data.λ; # how budget sensitive the consumer is
    B = data.B; # consumer's budget

    # build empty model -
    model = modeltype();
    model.m = m;
    model.γ = γ;
    model.σ = σ;
    model.Z = Z;
    model.C = C;
    model.λ = λ;
    model.B = B;

    # return -
    return model;
end