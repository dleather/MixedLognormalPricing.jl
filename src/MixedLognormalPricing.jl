module MixedLognormalPricing

# Write your package code here.
using Revise
using Parameters
using LinearAlgebra
using Random
using Statistics
using StaticArrays


#### Support for Markov Switch (MS)
@with_kw struct MS
	S::Int64 #Number of discrete regimes
	P::Matrix{Float64} #Transition matrix (rows sum to unity)
	@assert S==size(P)[1]
	@assert S==size(P)[2]

end

@with_kw struct MSVAR1
    N :: Int64
	ms :: MS #Markov switch
	S :: Int64 = ms.S
	
	μ :: Vector{Vector{Float64}}
	Φ :: Vector{Matrix{Float64}}
	Ω :: Vector{Matrix{Float64}}

	@assert length(μ) == S
	@assert length(Φ) == S
	@assert length(Ω) == S

	Σ :: Vector{Matrix{Float64}} = [cholesky(Ω[s]).L for s = 1:S]

end

# Generate a 1-step Markov Switch w/ initial state s0 and rng object
function draw_s(rng::R, s0::Int64, ms::MS) where R <: AbstractRNG
	@unpack P = ms
	u = rand(rng)
	out_s = findfirst(x -> x < u, cumsum(P[s0,:]))
	return out_s
end

# Above, but given a specific random draw, u
function draw_s_cond_shock(u::Float64, s0::Int64, ms::MS)
	@unpack P = ms
	out_s = findfirst(x -> x < u, cumsum(P[s0,:]))
	return out_s
end

# Generate a N-step Markov Switch w/ initial state s0 and rng object
function draw_s_path(rng::R, s0::Int64, ms::MS, N::Int64) where R <: AbstractRNG
	@unpack P = ms
	u_mat = rand(rng,N - 1)
	out_s_path = Vector{Int64}(undef, N)
	out_s_path[1] = s0
	csP = cumsum(P, dims = 2)
	for n = 1:N-1
        for s = 1:ms.S
            if csP[out_s_path[n], s] > u_mat[n]
                out_s_path[n+1] = s
                break
            end
        end
	end

	return out_s_path
end

# Generate Compute conditional expectation
function compute_Ex(s::Int64, Exm1::Vector{Float64}, μ::Vector{Vector{Float64}},
    Φ::Vector{Matrix{Float64}})

    # E_{t,n}[x_{t+n-1}] = μ(s_{t+n-1}) + Φ(s_{t+n-1}) E_{t,n-1}[x_{t+n-2}]
    out_Ex = μ[s] + Φ[s] * Exm1

    return out_Ex
end

#Given (sₜ, …, sₜ₊ₙ₋₁), compute [ Eₜ[x_t], Eₜ[x_{t+1}], ⋯ , Eₜ[x_{t+n-1}] ]
function compute_Ex_path(s_path::Vector{Int64}, x0::Vector{Float64},
    μ::Vector{Vector{Float64}}, Φ::Vector{Matrix{Float64}})

    N = length(s_path) - 1 #How far out do we propogate X
    m = length(x0) #Dimension of random vector

    Ex_path = [Vector{Float64}(undef, m) for n = 1:(N+1)]
    Ex_path[1] = x0

    for n=1:N
        Ex_path[n+1] = compute_Ex(s_path[n+1], Ex_path[n], μ, Φ)
    end

    # N x 1 Vector of (n x 1) vectors
    # [ Eₜ[x_t], Eₜ[x_{t+1}], ⋯ , Eₜ[x_{t+n-1}] ]
    return Ex_path

end

# Accumulates the conditional expectation of x_t -> x_t + ⋯ + xₜ₊ₙ₋₁
function compute_EΨ_path(Ex_path::Vector{Vector{Float64}})

	#E_{t,n}[Ψ_{t,n}] = ∑_{i=0}^{n-1} Eₜ[x_{t+i}]
	#[ Eₜ[Ψ₁], Eₜ[Ψ₂], ⋯ , Eₜ[Ψ_{N-1}] ]
	return cumsum(Ex_path)
end

# Iterator on conditional variance of xₜ₊ₙ
function compute_Vx(s::Int64, Vxm1::Matrix{Float64}, Φ::Vector{Matrix{Float64}},
    Ω::Vector{Matrix{Float64}})

    return Ω[s] + Φ[s] * Vxm1 * Φ[s]'

end


function compute_Vx_path(s_path::Vector{Int64}, Φ::Vector{Matrix{Float64}},
    Ω::Vector{Matrix{Float64}})

    N = length(s_path) - 1 #How far out do we propogate X
    m = size(Φ[1])[1] #Dimension of random vector

    Vx_path = [Matrix{Float64}(undef, m, m) for n=1:(N+1)]
    Vx_path[1] = zeros(m,m)

    for n=1:N
        Vx_path[n+1] = compute_Vx(s_path[n+1], Vx_path[n], Φ, Ω)
    end

    # [ Vₜ[xₜ], Vₜ[x_{t+1}], ⋯ , Vₜ[x_{t+n-1}] ]
    return Vx_path

end

# Accumulator of Φ
function compute_Φ̃(s::Int64, Φ̃m1::Matrix{Float64}, Φ::Vector{Matrix{Float64}})
	return Φ[s] * Φ̃m1
end

# Accumulator of Φ, unrolled and mutated
function compute_Φ̃!(Φ̃::Matrix{Float64}, s::Int64, Φ̃m1::Matrix{Float64},
                    Φ::Vector{Matrix{Float64}})

    n = size(Φ̃, 1)

    for n1 = 1:n
        for n2 = 1:n
            for k=1:n
                Φ̃[n1,n2] += Φ[s][n1,k] * Φ̃m1[k,n2]
            end
        end
    end

end

function compute_Φ̃_path(s_path::Vector{Int64}, Φ::Vector{Matrix{Float64}})

	N = length(s_path) - 1 #How far out do we propogate X
	m = size(Φ[1])[1] #Dimension of random vector
	
    Φ̃_path = Vector{Matrix{Float64}}(undef, N+1)
    for n = 1:N+1
        Φ̃_path[n] = zeros(m, m)
    end
	Φ̃_path[1] = Matrix(I, m, m)

	for n=1:N
		compute_Φ̃!(Φ̃_path[n+1], s_path[n+1], Φ̃_path[n], Φ)
	end

	# [ I , Φ(s_{t+1}), Φ(s_{t+2})Φ(s_{t+1}), ⋯v, Φ(s_{t+n-1}) ⋯ Φ(s_{t+1}) ]
	return Φ̃_path

end

function compute_Φ̃_path!(Φ̃_path,s_path,
    Φ::Vector{Matrix{Float64}})

    N = length(s_path) - 1 #How far out do we propogate X
    m = size(Φ[1])[1] #Dimension of random vector

    for mm = 1:m
        Φ̃_path[1][mm, mm] = 1.0
    end

    for n=1:N
        compute_Φ̃!(Φ̃_path[n+1], s_path[n+1], Φ̃_path[n], Φ)
    end

    # [ I , Φ(s_{t+1}), Φ(s_{t+2})Φ(s_{t+1}), ⋯, Φ(s_{t+n-1}) ⋯ Φ(s_{t+1}) ]
end

function compute_Φ̃_path_cov(s_path::Vector{Int64}, Φ::Vector{Matrix{Float64}})

    N = length(s_path) #How far out do we propogate X
    M = size(Φ[1])[1] #Dimension of random vector

    Φ̃_mat = [zeros(M,M) for n = 1:N, m = 1:N]

    for n = 1:N
        compute_Φ̃_path!(@view(Φ̃_mat[n,n:N]), @view(s_path[n:N]), Φ)
    end

    return Φ̃_mat

end

function compute_Φ̃_path_cov!(Φ̃_mat::Matrix{Matrix{Float64}},
    s_path::Vector{Int64}, Φ::Vector{Matrix{Float64}})

    N = length(s_path) #How far out do we propogate X

    for n = 1:N
        compute_Φ̃_path!(@view(Φ̃_mat[n,n:N]), s_path[n:N], Φ)
    end

end

function compute_Covjk(j::Int64, k::Int64, Vx_path::Vector{Matrix{Float64}}, 
    Φ̃_mat::Matrix{Matrix{Float64}})

    #return Vx_path[j+1] * (Φ̃_path[j+1] \ Φ̃_path[k+1])
    return Vx_path[j+1] * Φ̃_mat[j+1,k+1]

end

function compute_Covjk!(out_sum::Matrix{Float64}, out_cov::Matrix{Float64},
    m::Int64, j::Int64, k::Int64, Vx_path::Vector{Matrix{Float64}},
    Φ̃_mat::Vector{Matrix{Float64}})

    for i = 1:m
        for jj = 1:m
            out_cov[i, jj] = 0.0
            for kk = 1:m
                out_cov[i, jj] += Vx_path[j+1][i, kk] * Φ̃_mat[j+1,k+1][kk, jj]
            end
            out_sum[i, jj] += out_cov[i, jj]
        end
    end

end

function compute_sumcov_Ψ(Vx_path::Vector{Matrix{Float64}}, 
    Φ̃_mat::Vector{Matrix{Float64}})
    n = length(Vx_path) - 1
    m = size(Vx_path[1])[1]

    cov_sum = zeros(m, m)
    out_cov = zeros(m, m)

    for j=1:n-2
        for k=j:n-1
            compute_Covjk!(cov_sum, out_cov, m ,j, k, Vx_path, Φ̃_mat)
        end
    end

    return cov_sum + cov_sum'

end

function compute_covmat_Ψ(Vx_path::Vector{Matrix{Float64}}, 
    Φ̃_path::Matrix{Matrix{Float64}})

    n = length(Vx_path) - 1
    m = size(Vx_path[1])[1]
    cov_mat = [zeros(m,m) for i=1:n-1, j=1:n-1]

    for j=1:n-1
        for k=j+1:n
            cov_mat[j, k-1] = compute_Covjk(j, k, Vx_path, Φ̃_path)
        end
    end

    # (N-2) × (N-1) Matrix of (m x m) Matrices
    #[ COV_{1,2}, COV_{1,3}, ⋯ , COV_{1,N-1}  ]
    #[     ⋅    , COV{2,3},  ⋯ , COV_{2,N-1}  ]
    #[     ⋮    ,     ⋮    , ⋱ ,      ⋮       ]
    #[     ⋅    ,     ⋅    , ⋯ , COV{N-2,N-1} ]
    return cov_mat
end

function compute_VΨ(Vx_path::Vector{Matrix{Float64}}, 
    cov_mat::Matrix{Matrix{Float64}})

    N = length(Vx_path) - 1

    #Compute sum of  variance term - ∑_{i=0}^{n-1} Vₜ,ₙ[xₜ₊ₙ₋₁]
    cumsum_var = cumsum(Vx_path)
    m = size(Vx_path[1])[1]

    VΨ_out = similar(Vx_path)
    for n = 0:N 
        if n > 1
            VΨ_out[n+1] = cumsum_var[n+1]
            for j=1:n-1 
                for k=j+1:n
                    for mm = 1:m
                        for mmm = 1:m
                            VΨ_out[n+1][mm, mmm] += cov_mat[j, k-1][mm, mmm] +
                                cov_mat[j, k-1][mmm, mm]
                        end
                    end
                end
            end

        else
            VΨ_out[n+1] = cumsum_var[n+1]
        end
    end

    return VΨ_out

end

function compute_Eη_path(δ::Matrix{Float64}, EΨ_path::Vector{Vector{Float64}}, 
    VΨ_path::Vector{Matrix{Float64}})
    
    N = size(EΨ_path)[1] - 1
    Eη_path = Vector{Float64}(undef, N + 1)

    for n = 1:N+1
        Eη_path[n] = exp( δ * EΨ_path[n] + 0.5 * δ * VΨ_path[n] * δ' )[1]
    end

    return Eη_path

end

function simulate_msvar_cond_regime_path(rng::R, x0::Vector{Float64},
    spath::Vector{Int64}, msvar::MSVAR1) where R <: AbstractRNG 

    @unpack N, S, μ, Φ, Σ = msvar
    T = length(spath)

    x_mat = zeros(N, T)
    ϵ = randn(rng, N, T - 1)

    x_mat[:, 1] .= x0
    s = 0

    for t=2:T
        s = spath[t]
        for n = 1:N
            x_mat[n, t] += μ[s][n]
            for n2 = 1:N
                x_mat[n, t] += Φ[s][n,n2]*x_mat[n2,t-1] + Σ[s][n,n2]*ϵ[n2,t-1]
            end
        
        end
    end

    return x_mat

end

function compute_P_cond_xpath(s_path::Vector{Int64}, x0::Vector{Float64},
    δ::Matrix{Float64}, msvar::MSVAR1)

    @unpack μ, Φ, Ω = msvar
    # Compute E[xₜ | xₜ], E[xₜ + xₜ₊₁ | xₜ], ⋯, E[xₜ + ⋯ + xₜ₊ₙ₋₁ | xₜ]  
    Ex_path = compute_Ex_path(s_path, x0, μ, Φ)
    EΨ_path = compute_EΨ_path(Ex_path)

    # Compute V[xₜ | xₜ], V[xₜ + xₜ₊₁ | xₜ], ⋯, V[xₜ + ⋯ + xₜ₊ₙ₋₁ | xₜ]  
    Vx_path = compute_Vx_path(s_path, Φ, Ω)
    Φ̃_path = compute_Φ̃_path_cov(s_path, Φ)
    cov_mat = compute_covmat_Ψ(Vx_path, Φ̃_path)
    VΨ_path = compute_VΨ(Vx_path, cov_mat)

    # Compute E[ηₜ | xₜ], E[ηₜ * ηₜ₊₁ | xₜ], ⋯, E[ηₜ * ⋯ * ηₜ₊ₙ₋₁ | xₜ]
    Eη_path = compute_Eη_path(δ, EΨ_path, VΨ_path)

    price = sum(Eη_path)
    ηT = Eη_path[end]

    addOutputs = (Eη_path = Eη_path, EΨ_path = EΨ_path, VΨ_path = VΨ_path,
                  Ex_path = Ex_path, Vx_path = Vx_path, cov_mat = cov_mat,
                  Φ̃_path = Φ̃_path)
    return (price, ηT, addOutputs)
end

function compute_halfmc_price(rng::R, x0::Vector{Float64}, s0::Int64,
    N_paths::Int64, length_path::Int64, ms::MS, msvar::MSVAR1,
    δ::Matrix{Float64}) where R <: AbstractRNG

    @unpack μ, Φ, Ω, Σ = msvar

    # Generate paths
    s_paths = [Vector{Int64}(undef, length_path) for n = 1:N_paths]
    for n = 1:N_paths
        s_paths[n] = draw_s_path(rng, s0, ms, length_path)
    end

    price_mat = Vector{Float64}(undef, N_paths)
    ηT_mat = Vector{Float64}(undef, N_paths)
    Eηpath_mat = [Vector{Float64}(undef, length_path) for n = 1:N_paths]

    for (i,s_path) ∈ enumerate(s_paths)

        price, ηT, addOutputs =  compute_P_cond_xpath(s_path, x0,δ, msvar)

        # Compute the price
        price_mat[i] = price
        ηT_mat[i] = ηT
        Eηpath_mat[i] = addOutputs.Eη_path

    end

    mean_price = mean(price_mat)
    std_price = std(price_mat)

    addOutputs = (std_price = std_price, price_mat = price_mat, ηT_mat = ηT_mat,
                  Eηpath_mat = Eηpath_mat)

    return (mean_price, addOutputs)

end

function compute_mc_price(rng::R, x0::Vector{Float64}, s0::Int64,
    N_paths::Int64, length_path::Int64, ms::MS, msvar::MSVAR1,
    δ::Matrix{Float64}) where R <: AbstractRNG

    @unpack μ, Φ, Ω, Σ, N = msvar

    s_paths = [Vector{Int64}(undef, length_path) for n = 1:N_paths]
    x_paths =
        [Vector{Matrix{Float64}}(undef, N, length_paths) for n = 1:N_paths]
    Eη_paths = [Vector{Float64}(undef, length_paths) for n = 1:N_paths]
    for n = 1:N_paths
        s_paths[n][:] .= draw_s_path(rng, s0, ms, length_path)
        x_paths[n][:,:] .= simulate_msvar_cond_regime_path(rng, x0, s_path,
                            msvar)
        Eη_paths[n][:] .= cumsum(δ * x_paths[s][:, :])
        price_mat[n] = sum(Eη_paths[n])
    end

    mean_price = mean(price_mat)
    std_price = std(price_mat)

    addOutputs = (std_price = std_price, price_mat = price_mat,
                  Eη_paths = Eη_paths, x_paths = x_paths, s_paths = s_paths)
    
    return (mean_price, addOutputs)

end


export MS
export compute_Ex_path
export compute_EΨ_path
export compute_Vx_path
export compute_Φ̃_path
export compute_covmat_Ψ
export compute_Φ̃_path_cov
export compute_VΨ
export compute_Eη_path
export compute_halfmc_price
export MSVAR1
export compute_mc_price

end
