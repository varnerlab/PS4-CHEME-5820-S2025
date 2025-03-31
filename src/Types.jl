abstract type AbstractBanditAlgorithmModel end
abstract type AbstractBanditConsumerContextModel end

mutable struct MyEpsilonGreedyAlgorithmModel <: AbstractBanditAlgorithmModel

    # data -
    K::Int64 # number of arms in each category
    n::Array{Float64,1} # recomended quantity of each good to purchase

    # constructor -
    MyEpsilonGreedyAlgorithmModel() = new();
end


mutable struct MyBanditConsumerContextModel <: AbstractBanditConsumerContextModel

    # data -
    m::Int64 # number of of goods to choose from
    γ::Array{Int64,1} # consumer's preference for each category of goods
    σ::Array{Float64,1} # uncetainty of consumer's preference for each category of goods
    C::Array{Float64,1} # price of each good in each category
    λ::Float64 # how budget sensitive the consumer is
    B::Float64 # consumer's budget
    Z::Normal # consumer's preference for each category of goods

    # constructor -
    MyBanditConsumerContextModel() = new();
end