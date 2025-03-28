# PRIVATE METHODS BELOW HERE ================================================================================= #
# placeholder - always return 0
_null(action::Int64)::Int64 = return 0;

function _solve(model::MyExploreFirstAlgorithmModel; T::Int = 0, world::Function = _null, 
    context::MyBanditConsumerContextModel = nothing)::Array{Float64,2}

    # initialize -
    category_action_map = model.K # get the number of arms
    K = sum(values(category_action_map)); # number of arms (sum of all arms in each category)
    rewards = zeros(Float64, T, K); # rewards for each arm
    n = model.n; # get the recommended quantity of each good in each category
    number_of_categories = length(category_action_map); # number of categories
    
    # flatten the recommended quantity of each good in each category -
    goods = zeros(Int64, K); # recommended quantity of each good in each category
    counter = 1;
    for i ∈ 1:number_of_categories
        for j ∈ 1:category_action_map[i]
            goods[counter] = n[i][j]; # set the recommended quantity of each good in each category
            counter += 1; # increment the counter
        end
    end

    # how many expore steps should we take?
    Nₐ = ((T/K)^(2/3))*(log(T))^(1/3) |> x -> round(Int,x); # number of explore steps
    
    # exploration phase -
    counter = 1;
    for a ∈ 1:K

        # build action vector -
        av = zeros(Int64, K); # action vector
        av[a] = 1; # set the action to the current arm

        # quantity value -
        nv = zeros(Int64, K); # quantity vector
        nv[a] = goods[a]; # set the quantity to a random value

        for _ ∈ 1:Nₐ
            rewards[counter, a] = world(a, nv, context); # store from action a
            counter += 1;
        end
    end

    μ = zeros(Float64, K); # average reward for each arm
    for a ∈ 1:K
        μ[a] = findall(x -> x != 0.0, rewards[:, a]) |> i-> mean(rewards[i, a]); # compute the average reward
    end

    # exploitation phase -
    a = argmax(μ); # compute the arm with best average reward
    nv = zeros(Int64, K); # quantity vector
    for _ ∈ 1:(T - Nₐ*K)
        nv[a] = goods[a]; # set the purchased quantity to a random value
        rewards[counter, a] = world(a, nv, context); # store the reward
        counter += 1;
    end
    
    # return -
    return rewards;
end

function _solve(model::MyEpsilonGreedyAlgorithmModel; T::Int = 0, world::Function = _null, 
    context::MyBanditConsumerContextModel = nothing)::Array{Float64,2}

    # initialize -
    category_action_map = model.K # get the number of arms
    number_of_categories = length(category_action_map); # number of categories
    K = sum(values(category_action_map)); # number of arms (sum of all arms in each category)
    rewards = zeros(Float64, T, K); # rewards for each arm

    # flatten the recommended quantity of each good in each category -
    goods = zeros(Int64, K); # recommended quantity of each good in each category
    n = model.n; # get the recommended quantity of each good in each category
    number_of_categories = length(category_action_map); # number of categories
    counter = 1;
    for i ∈ 1:number_of_categories
        for j ∈ 1:category_action_map[i]
            goods[counter] = n[i][j]; # set the recommended quantity of each good in each category
            counter += 1; # increment the counter
        end
    end

    # main -
    for t ∈ 1:T
        ϵₜ = (1.0/(t^(1/3)))*(log(K*t))^(1/3); # compute the epsilon value -

        # if we were to purchase stuff, how much would we purchase?
        p = rand(); # role a random number
        aₜ = zeros(Int64, number_of_categories); # initialize action vector
        if (p ≤ ϵₜ)
            for i ∈ 1:number_of_categories
                aₜ[i] = rand(1:category_action_map[i]); # randomly select an arm
            end
        else
            
            # decide which arm to play *in each category* -
            for i ∈ 1:number_of_categories
                μ = zeros(Float64, category_action_map[i]); # average reward for each arm in this category
                for a ∈ 1:category_action_map[i]
                    μ[a] = findall(x -> x != 0.0, rewards[:, a]) |> k-> mean(rewards[k, a]); # compute the average reward
                end
                aₜ[i] = argmax(μ); # compute the arm with best average reward
            end            
        end

        rₜ = world(aₜ, goods, context); # get the reward from the world
        for i ∈ 1:number_of_categories
            j = 1; # what is the index of the arm in the rewards array? # TODO index?
            rewards[t, j] = rₜ 
        end 
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
# PUBLIC METHODS ABOVE HERE ================================================================================== #`