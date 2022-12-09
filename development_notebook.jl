using Revise
using Random
using MixedLognormalPricing
using Plots
using ProgressMeter
using BenchmarkTools

###############################################################################
# Set Code Parameters

## Set the seed and define rng object
seed_num = 1536734
rng = Xoshiro(seed_num)

## Set the parameters of the model

    #### Markov Switch
    P = [0.9 0.1; 0.3 0.7]
    S = 2
    ms = MS(P = P, S = S)


    #### MS-VAR(1)
    μ₁ = [0.1, -0.1]
    μ₂ = -μ₁
    μ = [μ₁, μ₂]

    Φ₁ = [0.9 0.0; 0.0 0.8]
    Φ₂ = 0.6 .* Φ₁
    Φ = [Φ₁, Φ₂]

    Ω₁ = [0.1 0.0; 0.0 0.05]
    Ω₂ = 2 .* Ω₁
    Ω = [Ω₁, Ω₂]

    N = 2
    msvar = MSVAR1(N = N, ms = ms, μ = μ, Φ = Φ, Ω = Ω)

    #### Pricing parameters
    δ = [-0.1 0.1]

## Simulation parameters
N_paths = 100
length_path = 500
s0 = 1
x0 = [0.0, 0.0]
N_sims = 30

mprice_mat = Vector{Float64}(undef, N_sims)
@showprogress for i = 1:N_sims
    # Run the Simulation
    mprice_mat[i], ~ = 
        compute_halfmc_price(rng, x0, s0, N_paths, length_path, ms, msvar, δ)
end


N_paths2 = 100^2
length_path2 = 500

mprice_mat2 = Vector{Float64}(undef, N_sims)
@showprogress for i = 1:N_sims
    # Run the Simulation
    mprice_mat2[i], ~ = 
        compute_mc_price(rng, x0, s0, N_paths2, length_path2, ms, msvar, δ)
end
#=
#Get very good simulation of truth

N_paths3 = 100000
length_path3 = 500

mprice_mat3 = Vector{Float64}(undef, N_sims)
true_price_mat = Vector{Float64}(undef, N_sims)
@showprogress for i = 1:N_sims
    # Run the Simulation
    true_price_mat[i], ~ = 
        compute_mc_price(rng, x0, s0, N_paths3, length_path3, ms, msvar, δ)
end

l = @layout [a; b; c]
x_min = minimum([minimum(mprice_mat), minimum(mprice_mat2),
    minimum(true_price_mat)])
x_max = maximum([maximum(mprice_mat), maximum(mprice_mat2),
    maximum(true_price_mat)])
=#

l = @layout [a; b]
x_min = minimum([minimum(mprice_mat), minimum(mprice_mat2)])
x_max = maximum([maximum(mprice_mat), maximum(mprice_mat2)])
h1 = histogram(mprice_mat, label = "Half MC")
vline!(h1, [mean(mprice_mat)], label = "Half MC")
xlims!(h1, (x_min, x_max))
title!(h1,"Half MC")
h2 = histogram(mprice_mat2, label = "Full MC")
vline!(h2, [mean(mprice_mat2)], label = "Full MC")
xlims!(h2, (x_min, x_max))
title!(h2, "Full MC")
#=
h3 = histogram(true_price_mat, label = "True")
vline!(h3, [mean(true_price_mat)], label = "True")
xlims!(h3, (x_min, x_max))
title!(h3, "True")

fig1 = plot(h1, h2, h3, layout = l)
=#
fig1 = plot(h1, h2, layout = l)

seed_num = 1536734
rng = Xoshiro(seed_num)


mprice_mat_lse = Vector{Float64}(undef, N_sims)
@showprogress for i = 1:N_sims
    # Run the Simulation
    mprice_mat_lse[i], ~ = 
        compute_halfmc_price_lse(rng, x0, s0, N_paths, length_path, ms, msvar, δ)
end


mprice_mat2_lse = Vector{Float64}(undef, N_sims)
@showprogress for i = 1:N_sims
    # Run the Simulation
    mprice_mat2_lse[i], ~ = 
        compute_mc_price_lse(rng, x0, s0, N_paths2, length_path2, ms, msvar, δ)
end

#Get very good simulation of truth
#=
mprice_mat3_lse = Vector{Float64}(undef, N_sims)
true_price_mat_lse = Vector{Float64}(undef, N_sims)
@showprogress for i = 1:N_sims
    # Run the Simulation
    true_price_mat_lse[i], ~ = 
        compute_mc_price_lse(rng, x0, s0, N_paths3, length_path3, ms, msvar, δ)
end

l = @layout [a; b; c]
x_min = minimum([minimum(mprice_mat_lse), minimum(mprice_mat2_lse),
    minimum(true_price_mat_lse)])
x_max = maximum([maximum(mprice_mat_lse), maximum(mprice_mat2_lse),
    maximum(true_price_mat_lse)])
=#
x_min = minimum([minimum(mprice_mat_lse), minimum(mprice_mat2_lse)])
x_max = maximum([maximum(mprice_mat_lse), maximum(mprice_mat2_lse)])
h1_lse = histogram(mprice_mat_lse, label = "Half MC")
vline!(h1_lse, [mean(mprice_mat_lse)], label = "Half MC")
xlims!(h1_lse, (x_min, x_max))
title!(h1_lse,"Half MC")
h2_lse = histogram(mprice_mat2_lse, label = "Full MC")
vline!(h2_lse, [mean(mprice_mat2_lse)], label = "Full MC")
xlims!(h2_lse, (x_min, x_max))
title!(h2_lse, "Full MC")
#=
h3_lse = histogram(true_price_mat_lse, label = "True")
vline!(h3_lse, [mean(true_price_mat_lse)], label = "True")
xlims!(h3_lse, (x_min, x_max))
title!(h3_lse, "True")
fig1_lse = plot(h1, h2, h3, layout = l)
=#
fig1_lse = plot(h1, h2, layout = l)


compute_halfmc_price_lse(rng, x0, s0, N_paths, length_path, ms, msvar, δ)

@btime compute_halfmc_price_lse(rng, x0, s0, N_paths, length_path, ms, msvar, δ)

using Distributions

function draw_p(rng::R) where R <: AbstractRNG
    μ₁_mean = [0.1, -0.1]
    μ₂_mean = -μ₁_mean

    μ1_std = μ₁_mean
    μ2_std = μ₂_mean

    Φ₁_mean = [0.9 0.0; 0.0 0.8]
    Φ₂_mean = 0.6 .* Φ₁
    Φ₁_std = 0.5 * Φ₁_mean
    Φ₂_std = 0.5 * Φ₂_mean

    Ω₁_mean = [0.1 0.0; 0.0 0.05]
    Ω₂_mean = 2 .* Ω₁

    μ₁ = μ₁_mean + μ1_std .* randn(rng, 2)
    μ₂ = μ₂_mean + μ2_std .* randn(rng, 2)

    μ = [μ₁, μ₂]

    Φ₁ = Φ₁_mean + Φ₁_std .* randn(rng, 2, 2)
    Φ₂ = Φ₂_mean + Φ₂_std .* randn(rng, 2, 2)

    Φ = [Φ₁, Φ₂]

    dOmega1 = Wishart(10, Ω₁_mean)
    dOmega2 = Wishart(10, Ω₂_mean)

    Ω₁ = rand(rng, dOmega1)
    Ω₂ = rand(rng, dOmega2)
    Ω = [Ω₁, Ω₂]

    trig = 0
    if maximum(maximum([abs.(eigvals(Φ₁)), abs.(eigvals(Φ₂))])) > 1
        trig = 1
    end

    return μ, Φ, Ω
end