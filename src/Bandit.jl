# PRIVATE METHODS BELOW HERE ================================================================================= #
# placeholder - always return 0
_null(action::Int64)::Int64 = return 0;

"""
    private logic called by the public solve method. This implementation is similar to the `L7b` impl, but
    we've modified it to work with combinations, and use a weighted online average for the rewards.
"""
function _solve(model::MyEpsilonGreedyAlgorithmModel; T::Int = 0, world::Function = _null, 
    context::MyBanditConsumerContextModel = nothing)::Array{Float64,2}

    # initialize -
    K = model.K; # get the number of goods to choose from
    goods = model.n; # get the recommended quantity of each good in each category
    N = 2^K; # this is the maximum number of arms we can have (we have K goods, with each good being {0 | 1})
    rewards = Array{Float64,2}(undef, T, N); # rewards for possible arm
    μ = zeros(Float64, N); # average reward for each possible goods combination

    # initialiize the rewards to zero -
    fill!(rewards, 0.0); # fill the rewards array with zeros
    
    # main -
    for t ∈ 1:T
        ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -

        # if we were to purchase stuff, how much would we purchase?
        p = rand(); # role a random number
        aₜ = nothing; # initialize action vector
        î = nothing; # index of the combination of goods
        if (p ≤ ϵₜ)

            # which combination of goods to choose?
            î = rand(1:N); # randomly select an integer from 1 to N (this will be used to generate a binary representation of the action vector)
            aₜ = digits(î, base=2, pad=K); # generate a binary representation of the number, with K digits
        else
            î = argmax(μ); # compute the arm with best average reward
            aₜ = digits(î, base=2, pad=K); # generate a binary representation of the number, with K digits      
        end

        # call out to the world, record the result.
        rₜ = world(aₜ, goods, context); # get the reward from the world
        μ[î]+=0.8*(rₜ + μ[î]); # update the average reward for the chosen arm (learning rate = 0.8)
        rewards[t, î] = rₜ # store the reward
    end

    # return -
    return rewards;
end


# PRIVATE METHODS ABOVE HERE ================================================================================= #

# PUBLIC METHODS BELOW HERE ================================================================================== #`
"""
    solve(model::AbstractBanditAlgorithmModel; T::Int = 0, world::Function = _null)

Solve the bandit problem using the given model. 

### Arguments
- `model::AbstractBanditAlgorithmModel`: The model to use to solve the bandit problem.
- `T::Int = 0`: The number of rounds to play. Default is 0.
- `world::Function = _null`: The function that returns the reward for a given action. Default is the private `_null` function.

### Returns
- `Array{Float64,2}`: The rewards for each arm at each round.
"""
function solve(model::AbstractBanditAlgorithmModel; T::Int = 0, world::Function = _null, 
    context::MyBanditConsumerContextModel = nothing)::Array{Float64,2}
    return _solve(model, T = T, world = world, context = context);
end

"""
    regret(rewards::Array{Float64,2})::Array{Float64,1}

Compute the regret for the given rewards.

### Arguments
- `rewards::Array{Float64,2}`: The rewards for each arm at each round.

### Returns
- `Array{Float64,1}`: The regret at each round.
"""
function regret(rewards::Array{Float64,2})::Array{Float64,1}
    
    # initialize -
    T = size(rewards, 1); # how many rounds did we play?
    K = size(rewards, 2); # how many arms do we have?
    regret = zeros(Float64, T); # initialize the regret array

    # first: compute the best arm in hindsight -
    μ = zeros(Float64, K); # average reward for each arm
    for a ∈ 1:K
        μ[a] = findall(x -> x != 0.0, rewards[:, a]) |> i-> mean(rewards[i, a]); # compute the average reward
    end
    μₒ = maximum(μ); # compute the best average reward

    # compute the regret -
    for t ∈ 1:T

        # what action was taken at time t?
        tmp = 0.0;
        for j = 1:t
            aₜ = argmax(rewards[j, :]); # get the action that was taken
            tmp += μ[aₜ]; # compute the hypothetical average reward
        end
        regret[t] = μₒ*t - tmp; # compute the regret at time t
    end

    # return -
    return regret;
end
# PUBLIC METHODS ABOVE HERE ================================================================================== #