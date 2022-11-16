using Revise
using Random
using MixedLognormalPricing
using Plots
using ProgressMeter
using BenchmarkTools

seed_num = 1536734
P = [0.9 1.0-0.9; 0.3 0.7]
S = 2
ms = MS(P = P, S = S)
s0 = 1


μ₁ = [0.1, -0.1]
μ₂ = -μ₁
μ = [μ₁, μ₂]

Φ₁ = [0.9 0.0; 0.0 0.8]
Φ₂ = 0.6 .* Φ₁
Φ = [Φ₁, Φ₂]

s_path = [1, 2, 1, 1, 2]

x0 = [0.0, 0.0]

Ex_path = compute_Ex_path(s_path, x0, μ, Φ)
EΨ_path = compute_EΨ_path(Ex_path)

Ω₁ = [0.1 0.0; 0.0 0.05]
Ω₂ = 2 .* Ω₁
Ω = [Ω₁, Ω₂]

Vx_path = compute_Vx_path(s_path, Φ, Ω)

Φ̃_path = compute_Φ̃_path_cov(s_path, Φ)

cov_mat = compute_covmat_Ψ(Vx_path, Φ̃_path)

VΨ_path = compute_VΨ(Vx_path, cov_mat)

δ = [0.1 -0.2]


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
N_paths = 30
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

histogram(mprice_mat)

N_paths2 = 30^2
length_path2 = 500

mprice_mat2 = Vector{Float64}(undef, N_sims)
for i = 1:N_sims
    # Run the Simulation
    mprice_mat2[i], ~ = 
        compute_mc_price(rng, x0, s0, N_paths2, length_path2, ms, msvar, δ)
end