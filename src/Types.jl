abstract type AbstractBanditAlgorithmModel end
abstract type AbstractBanditConsumerContextModel end

mutable struct MyBanditConsumerContextModel <: AbstractBanditConsumerContextModel

    # data -
    m::Int64 # number of categories of goods
    γ::Dict{Int64, Array{Int64,1}} # consumer's preference for each category of goods
    σ::Dict{Int64, Array{Float64,1}} # uncetainty of consumer's preference for each category of goods
    β::Dict{Int64, Array{Beta,1}} # consumer's preference for each category of goods
    C::Dict{Int64, Array{Float64,1}} # price of each good in each category
    λ::Float64 # how budget sensitive the consumer is
    B::Float64 # consumer's budget

    # constructor -
    MyBanditConsumerContextModel() = new();
end

mutable struct MyExploreFirstAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Dict{Int64,Int64} # number of arms in each category

    # constructor -
    MyExploreFirstAlgorithmModel() = new();
end

mutable struct MyEpsilonGreedyAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Dict{Int64,Int64} # number of arms in each category

    # constructor -
    MyEpsilonGreedyAlgorithmModel() = new();
end

mutable struct MyUCB1AlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Dict{Int64,Int64} # number of arms in each category

    # constructor -
    MyUCB1AlgorithmModel() = new();
end
