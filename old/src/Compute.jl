function table(results::Dict{Int64, Matrix{Float64}}, 
    algorithm::MyEpsilonGreedyAlgorithmModel,
    context::MyBanditConsumerContextModel)::DataFrame

    # initialize -
    df = DataFrame();
    number_of_categories = context.m; # number of categories
    category_action_map = algorithm.K # get the number of arms
    σ = context.σ; # noise in utility calculation (unknown to bandits)
    B = context.B; # max budget (unknown to bandits)
    
    # loop over the categories -
    BC = 0.0;
    U = 0.0;
    for i ∈ 1:number_of_categories
    
        # Data for this categorty
        K = category_action_map[i]; # get the number of arms for this category
        data = results[i]; # get the data for this category

        μ = Array{Float64,1}(undef, K); # mean of the data
        for j ∈ 1:K
            μ[j] = filter(x -> x != 0.0, data[:,j]) |> x-> mean(x)
        end
       
        # which action should we take?
        aᵢ = argmax(μ); # this is which good to purchase in category i -
        BC += algorithm.n[i][aᵢ]*context.C[i][aᵢ]; # budget constraint
        U += (context.γ[i][aᵢ])*algorithm.n[i][aᵢ] # update the utility

        row_df = (
            category = i,
            action = aᵢ,
            γᵢ = context.γ[i][aᵢ], # preference of good in category i
            n = algorithm.n[i][aᵢ], # this is how much of good i to purchase (must be geq ϵ)
            unitcost = context.C[i][aᵢ], # cost of chosen good in category i
            cumspend = BC, # this is how much we spent
            remaining = B - BC, # budget constraint
            U = U, # utility w/o noise
        );
        
        # add the row to the dataframe -
        push!(df, row_df);
    end

    return df; # return the dataframe
end