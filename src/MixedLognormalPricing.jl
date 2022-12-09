module MixedLognormalPricing

# Write your package code here.
using Revise
using Parameters
using LinearAlgebra
using Random
using Statistics
using LogExpFunctions
using LeatherMarkovChain
using LeatherMSVAR

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

function compute_logEη_path(δ::Matrix{Float64}, EΨ_path::Vector{Vector{Float64}}, 
    VΨ_path::Vector{Matrix{Float64}})
    
    N = size(EΨ_path)[1] - 1
    logEη_path = Vector{Float64}(undef, N + 1)

    for n = 1:N+1
        logEη_path[n] =  (δ * EΨ_path[n] + 0.5 * δ * VΨ_path[n] * δ')[1]
    end

    return logEη_path

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


function compute_P_cond_xpath_lse(s_path::Vector{Int64}, x0::Vector{Float64},
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
    logEη_path = compute_logEη_path(δ, EΨ_path, VΨ_path)

    price = exp(logsumexp(logEη_path))
    ηT = exp(logEη_path[end])

    addOutputs = (logEη_path = logEη_path, EΨ_path = EΨ_path, VΨ_path = VΨ_path,
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

    for (i, s_path) ∈ enumerate(s_paths)

        price, ηT, addOutputs =  compute_P_cond_xpath(s_path, x0,δ, msvar)

        # Compute the price
        price_mat[i] = price
        ηT_mat[i] = ηT
        Eηpath_mat[i] = addOutputs.Eη_path

    end

    mean_price = mean(price_mat)
    #std_price = std(price_mat)

    #addOutputs = (std_price = std_price, price_mat = price_mat, ηT_mat = ηT_mat,
    #              Eηpath_mat = Eηpath_mat)

    return mean_price

end


function compute_halfmc_price_lse(rng::R, x0::Vector{Float64}, s0::Int64,
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
    logEηpath_mat = [Vector{Float64}(undef, length_path) for n = 1:N_paths]

    for (i,s_path) ∈ enumerate(s_paths)

        price, ηT, addOutputs =  compute_P_cond_xpath_lse(s_path, x0,δ, msvar)

        # Compute the price
        price_mat[i] = price
        ηT_mat[i] = ηT
        logEηpath_mat[i] = addOutputs.logEη_path

    end

    mean_price = mean(price_mat)
    #std_price = std(price_mat)

    ##addOutputs = (std_price = std_price, price_mat = price_mat, ηT_mat = ηT_mat,
     #             logEηpath_mat = logEηpath_mat)

    return mean_price

end

function compute_mc_price(rng::R, x0::Vector{Float64}, s0::Int64,
    N_paths::Int64, length_path::Int64, ms::MS, msvar::MSVAR1,
    δ::Matrix{Float64}) where R <: AbstractRNG

    @unpack μ, Φ, Ω, Σ, N = msvar

    s_paths = [Vector{Int64}(undef, length_path) for n = 1:N_paths]
    x_paths = [Matrix{Float64}(undef, N, length_path) for n = 1:N_paths]
    Eη_paths = [Vector{Float64}(undef, length_path) for n = 1:N_paths]
    price_mat = Vector{Float64}(undef, N_paths)
    for n = 1:N_paths
        s_paths[n][:] .= draw_s_path(rng, s0, ms, length_path)
        x_paths[n][:,:] .= simulate_msvar_cond_regime_path(rng, x0, s_paths[n],
                            msvar)
        Eη_paths[n][:] .= vec(exp.(cumsum(δ * x_paths[n], dims = 2)))
        price_mat[n] = sum(Eη_paths[n])
    end

    mean_price = mean(price_mat)
    #std_price = std(price_mat)

    #addOutputs = (std_price = std_price, price_mat = price_mat,
    #              Eη_paths = Eη_paths, x_paths = x_paths, s_paths = s_paths)
    
    return mean_price

end

function compute_mc_price_lse(rng::R, x0::Vector{Float64}, s0::Int64,
    N_paths::Int64, length_path::Int64, ms::MS, msvar::MSVAR1,
    δ::Matrix{Float64}) where R <: AbstractRNG

    @unpack μ, Φ, Ω, Σ, N = msvar

    s_paths = [Vector{Int64}(undef, length_path) for n = 1:N_paths]
    x_paths = [Matrix{Float64}(undef, N, length_path) for n = 1:N_paths]
    Eη_paths = [Vector{Float64}(undef, length_path) for n = 1:N_paths]
    price_mat = Vector{Float64}(undef, N_paths)
    for n = 1:N_paths
        s_paths[n][:] .= draw_s_path(rng, s0, ms, length_path)
        x_paths[n][:,:] .= simulate_msvar_cond_regime_path(rng, x0, s_paths[n],
                            msvar)
        Eη_paths[n][:] .= vec(cumsum(δ * x_paths[n], dims = 2))
        price_mat[n] = exp(logsumexp(Eη_paths[n]))
    end

    mean_price = mean(price_mat)
    #std_price = std(price_mat)

    #addOutputs = (std_price = std_price, price_mat = price_mat,
    #              Eη_paths = Eη_paths, x_paths = x_paths, s_paths = s_paths)
    
    return mean_price

end

# Bianchi (Journal of Econometrics ,2015) Proposition 1
function update_qtil(qtilm1::Vector{Float64}, Ωtil::Matrix{Float64})
        return Ωtil * qtilm1
end

 

function construct_Ω(msvar::MSVAR1)
    # Ω = bdiag(A₁,\ldots,Aₙ) × (H ⊗ Iₙ)
    @unpack Φ, N, ms = msvar
    @unpack P, S = ms

    bdiagΦ = zeros(N*S, N*S)
    for i in 1:S
            bdiagΦ[(i-1)*N+1:i*N, (i-1)*N+1:i*N] = Φ[i]
    end

    return bdiagΦ * kron(P', Matrix(I, N, N))
end

function construct_C(msvar::MSVAR1)
    # C = bdiag(C₁,\ldots,Cₙ)
    @unpack μ, N, ms = msvar
    @unpack S = ms

    C = zeros(N*S, S)
    for s = 1:S
        C[(s-1)*N+1:s*N, s] .= μ[s]
    end
    return C

end

function construct_Ωtil(msvar::MSVAR1)

    @unpack ms, N  = msvar
    @unpack S, P = ms
    H = P'

    Ω = construct_Ω(msvar)
    C = construct_C(msvar)

    Ωtil = zeros(Float64, M * N + M, M * N + M)
    Ωtil[1:NM, 1:NM] .= Ω
    Ωtil[1:NM, NM+1:end] .= C * H
    Ωtil[NM+1:end, NM+1:end] .= H

end

# Bianchi, Journal of Econometrics, 2015, Proposition 2
function construct_AAhat(msvar::MSVAR1)
    #  AAHat = bdiag(A₁ ⊗ A₁,…,Aₘ ⊗ Aₘ)
    @unpack Φ, N, ms = msvar
    @unpack S = ms

    AAhat = zeros(Float64, N * N * S, N * N * S)
    for s = 1:S
        AAhat[(s-1)*N*N+1:s*N*N, (s-1)*N*N+1:s*N*N] .= kron(Φ[s],Φ[s])
    end
    
    return AAhat
end

function construct_Ξ(msvar::MSVAR1)
    # Ξ = bdiag(A₁ ⊗ A₁,…,Aₘ ⊗ Aₘ) * (H ⊗ Iₙₙ)
    @unpack ms, N  = msvar
    @unpack S, P = ms
    H = P'

    AAhat = construct_AAhat(msvar)

    return AAhat * kron(H, Matrix(I, N * N, N * N))
end

function construct_VVhat(msvar::MSVAR1)
    # VVHat = bdiag((V₁ ⊗ V₁) * vec(Iₙ),…,(Vₘ ⊗ Vₘ)* vec(Iₙ))
    @unpack Σ, N, ms = msvar
    @unpack S = ms

    vecIn = Matrix(I, N, N)[:]

    VVhat = zeros(Float64, N * N * S, S)
    for s = 1:S
        VVhat[(s-1)*N*N+1:s*N*N, s] .= kron(Σ[s],Σ[s]) * vecIn
    end
    
    return VVhat
end

function construct_cchat(msvar::MSVAR1)
    # ccHat = bdiag(c₁ ⊗ c₁,…,cₘ ⊗ cₘ)

    @unpack μ, N, ms = msvar
    @unpack S = ms

    cchat = zeros(Float64, N * N * S, S)
    for s = 1:S
        cchat[(s-1)*N*N+1:s*N*N, s] .= kron(μ[s],μ[s])
    end

    return cchat
end

function construct_Vchat(msvar::MSVAR1)
    # Vchat = [hatVV + hatcc]
    VVhat = construct_VVhat(msvar)
    cchat = construct_cchat(msvar)

    return VVhat + cchat
end

function compute_DAChat(msvar::MSVAR1)
    # DACHat = bdiag(A₁ ⊗ c₁ + c₁ ⊗ A₁,…,Aₘ ⊗ cₘ + cₘ ⊗ Aₘ) (H ⊗ Iₙ)
    @unpack Φ, μ, N, ms = msvar
    @unpack S, P = ms
    H = P'

    bdiagDACHat = zeros(Float64, S * N * N, S * N)
    for s = 1:S
        bdiagDACHat[(s-1)*N*N+1:s*N*N, (s-1)*N+1:s*N] .= 
            kron(Φ[s],μ[s]) + kron(μ[s],Φ[s])
    end

    return bdiagDACHat * kron(H, Matrix(I, N, N))

    
end

function compute_Ξtilde(msvar::MSVAR1)

    @unpack ms, N  = msvar
    @unpack S, P = ms

    DAChat = compute_DAChat(msvar)
    Ξ = construct_Ξ(msvar)
    Vchat = construct_Vchat(msvar)
    Ωtil = construct_Ωtil(msvar)
    H = P'

    MNN = S * N * N
    MN = S * N

    Ξtil = zeros(Float64, MNN + MN + S, MNN + MN + S)
    Ξtil[1:MNN, 1:MNN] .= Ξ
    Ξtil[1:MNN, MNN+1:MNN+MN] .= DAChat
    Ξtil[1:MNN, MNN+MN+1:MNN+MN+S] .= Vchat * H
    Ξtil[MNN+1:end, MNN+1:end] .= Ωtil

    return Ξtil

end

## First Conditional Moments - Leather
# ̄qₜₙʲ = Eₜ[Xₜ₊ₙ₋₁ | sₜ₊ₙ₋₁ = j]
# ̄qₜₙ = [qₜₙ¹′,…,qₜₙˢ′]′ = ̂μ +  \hat{BQ} ̄qₜₙ₋₁

function compute_̂μ(msvar::MSVAR1)
    @unpack μ, S , N = msvar

    out = zeros(Float64, N * S)

    @inbounds for s = 1:S
        out[(s-1)*N+1:s*N] .= μ[s]
    end

    return out
end

function compute_BΦ(msvar::MSVAR1)

    @unpack Φ, S, N, ms = msvar
    @unpack B = ms

    bdiag_mat = zeros(Float64, S * N, S * N)
    @inbounds for s = 1:S
        bdiag_mat[(s-1)*N+1:s*N, (s-1)*N+1:s*N] .= Φ[s]
    end

    bΦ = bdiag_mat * kron(B, Matrix(I, N, N))

    return bΦ
end

function compute_̄q_path(M::Int64, μ::Vector{Float64}, BΦ::Matrix{Float64}, 
            q0::Vector{Float64})
    
    SN = length(μ)

    q_path = zeros(Float64, SN, M)
    q_path[:, 1] .= q0

    @inbounds for m = 2:M
        q_path[:, m] .= μ + BΦ * q_path[:, m-1]
    end

    return q_path

end

function compute_̄a_path(M::Int64, μ::Vector{Float64}, BΦ::Matrix{Float64}, 
    q0::Vector{Float64}, msvar::MSVAR1)

    @unpack ms, S, N = msvar
    @unpack B = ms

    μtil = [μ; μ]
    BΦtil = zeros(2*N*S,2*N*S)
    BΦtil[1:N*S, 1:N*S] .= kron(B, Matrix(I, N, N))
    BΦtil[1:N*S, N*S+1:end] .= BΦ
    BΦtil[N*S+1:end, N*S+1:end] .= BΦ

    a0 = [q0; q0]

    a_path = zeros(Float64,2*S*N, M)
    a_path[:, 1] .= a0

    @inbounds for m = 2:M
        a_path[:, m] .= μtil #+ BΦtil * a_path[:, m-1]
        @inbounds for n1 = 1:N*S*2
            @inbounds for n2 = 1:N*S*2
                a_path[n1, m] += BΦtil[n1, n2] * a_path[n2, m-1]
            end
        end
    end

    return a_path

end

function compute_Ξ(msvar::MSVAR1)
    @unpack ms, N, Φ  = msvar
    @unpack S, B = ms

    bdiag_mat = zeros(Float64, S * N * N, S * N * N)
    @inbounds for s = 1:S
        bdiag_mat[(s-1)*N*N+1:s*N*N, (s-1)*N*N+1:s*N*N] .= kron(Φ[s],Φ[s])
    end

    return bdiag_mat * kron(B, Matrix(I, N * N, N * N))
end

function compute_BΦμ(msvar::MSVAR1)
    @unpack ms, N, Φ, μ  = msvar
    @unpack S, B = ms

    bdiag_mat = zeros(Float64, S * N * N, S * N)
    @inbounds for s = 1:S
        bdiag_mat[(s-1)*N*N+1:s*N*N, (s-1)*N+1:s*N] .= kron(Φ[s],μ[s]) + 
            kron(μ[s],Φ[s])
    end

    return bdiag_mat * kron(B, Matrix(I, N, N))
end

function compute_μμΣΣ(msvar::MSVAR1)

    @unpack ms, N, μ, Σ  = msvar
    @unpack S, B = ms

    μμ = zeros(S * N * N)
    ΣΣ = zeros(S * N * N)
    @inbounds for s = 1:S
        μμ[(s-1)*N*N+1:s*N*N] .= (μ[s] * μ[s]')[:]
        ΣΣ[(s-1)*N*N+1:s*N*N] .= (Σ[s] * Σ[s]')[:]
    end

    return μμ + ΣΣ
end

function compute_̄Q_path!(Q_path::Matrix{Float64}, M::Int64,
    q_path::Matrix{Float64}, Q0::Vector{Float64}, Ξ::Matrix{Float64},
    BΦμ::Matrix{Float64}, μμΣΣ::Vector{Float64}, msvar::MSVAR1)

    @unpack S, N = msvar

    Q_path[:, 1] .= Q0

    @inbounds for m = 2:M
        #Q_path[:, m] .= Ξ * Q_path[:, m-1] + BΦμ * q_path[:, m-1] + μμΣΣ
        @inbounds for n1 = 1:N*S*S
            Q_path[n1,m] = μμΣΣ[n1]
            @inbounds for n2 = 1:N*S*S
                Q_path[n1, m] += Ξ[n1, n2] * Q_path[n2, m-1]
            end
            @inbounds for n2 = 1:N*S
                Q_path[n1, m] += BΦμ[n1, n2] * q_path[n2, m-1]
            end
        end
        
    
    end

end

function compute_Iμ(msvar::MSVAR1)
    @unpack ms, N, μ, S  = msvar

    Iμ = zeros(S * N * N, S * N)
    @inbounds for s = 1:S
        Iμ[(s-1)*N*N+1:s*N*N, (s-1)*N+1:s*N] .= kron(Matrix(I,N,N),μ[s])
    end

    return Iμ
end

function compute_μI(msvar::MSVAR1)
    @unpack ms, N, μ, S  = msvar

    μI = zeros(S * N * N, S * N)
    @inbounds for s = 1:S
        μI[(s-1)*N*N+1:s*N*N, (s-1)*N+1:s*N] .= kron(μ[s], Matrix(I,N,N))
    end

    return μI
end

function compute_BIΦ(msvar::MSVAR1)

    @unpack ms, N, Φ, S  = msvar
    @unpack B = ms

    bdiag_mat = zeros(S*N*N, S*N*N)
    @inbounds for s = 1:S
        bdiag_mat[(s-1)*N*N+1:s*N*N, (s-1)*N*N+1:s*N*N] .= 
            kron(Matrix(I,N,N), Φ[s])
    end

    return bdiag_mat * kron(B, Matrix(I, N*N, N*N))
end

function compute_BΦI(msvar::MSVAR1)

    @unpack ms, N, Φ, S  = msvar
    @unpack B = ms

    bdiag_mat = zeros(S*N*N, S*N*N)
    @inbounds for s = 1:S
        bdiag_mat[(s-1)*N*N+1:s*N*N, (s-1)*N*N+1:s*N*N] .= 
            kron(Φ[s], Matrix(I,N,N))
    end

    return bdiag_mat * kron(B, Matrix(I, N*N, N*N))
end

function compute_̄A_path(M::Int64, q_path::Matrix{Float64},
        Q_path::Matrix{Float64}, msvar::MSVAR1, Iμ::Matrix{Float64},
        μI::Matrix{Float64}, BIΦ::Matrix{Float64}, BΦI::Matrix{Float64})

        @unpack S, N, ms = msvar
        @unpack B = ms

        A_path = zeros(Float64, S * N * N, M)
        A_path[:,1] .= zeros(S * N * N)

        # Matrix that loads on to previous period's A
        BIn = kron(B, Matrix(I,N,N))
        BIn2 = kron(B, Matrix(I,N * N,N * N))

        bR_mat = [BIΦ Iμ; zeros(S*N, S*N*N) BIn]
        bL_mat = [BΦI μI; zeros(S*N, S*N*N) BIn]

        # autocov mat
        acL_mat = Array{Float64}(undef, S*N*N + S*N, M, M)
        acR_mat = Array{Float64}(undef, S*N*N + S*N, M, M)
        @inbounds for m = 1:M
            acL_mat[1:S*N*N,m,m] = @view(Q_path[:,m])
            acR_mat[1:S*N*N,m,m] = @view(Q_path[:,m])
            acL_mat[S*N*N+1:end,m,m] = @view(q_path[:,m])
            acR_mat[S*N*N+1:end,m,m] = @view(q_path[:,m])
        end

        @inbounds for m = 2:M
            #A_path[:, m] .= BIn2 * A_path[:, m-1] .+ Q_path[:, m]
            @inbounds for n1 = 1:S*N*N
                @inbounds for k = 1:S*N*N
                    A_path[n1, m] += BIn2[n1, k] * A_path[k, m-1]
                end
                A_path[n1, m] += Q_path[n1, m]
            end
            @inbounds for n = m-1:-1:1
                #acL_mat[:,n,m] .= bL_mat * @view(acL_mat[:,n,m-1])
                #acR_mat[:,n,m] .= bR_mat * @view(acR_mat[:,n,m-1])
                @inbounds for n1 = 1:S*N*N
                    @inbounds for k = 1:S*N*N
                        acL_mat[n1, n, m] += bL_mat[n1, k] * acL_mat[k, n, m-1]
                        acR_mat[n1, n, m] += bR_mat[n1, k] * acR_mat[k, n, m-1]
                    end
                end
                @inbounds for k = 1:S*N*N
                    A_path[k, m] += acL_mat[k,n,m]
                    A_path[k, m] += acR_mat[k,n,m]
                end
            end
        end

    return A_path
end

function compute_̄A_path!(acL_mat::Array{Float64}, acR_mat::Array{Float64}, 
    M::Int64, q_path::Matrix{Float64},
    Q_path::Matrix{Float64}, msvar::MSVAR1, Iμ::Matrix{Float64},
    μI::Matrix{Float64}, BIΦ::Matrix{Float64}, BΦI::Matrix{Float64})

    @unpack S, N, ms = msvar
    @unpack B = ms

    A_path = zeros(Float64, S * N * N, M)
    A_path[:,1] .= zeros(S * N * N)

    # Matrix that loads on to previous period's A
    BIn = kron(B, Matrix(I,N,N))
    BIn2 = kron(B, Matrix(I,N * N,N * N))

    bR_mat = [BIΦ Iμ; zeros(S*N, S*N*N) BIn]
    bL_mat = [BΦI μI; zeros(S*N, S*N*N) BIn]

    # autocov mat
    #acL_mat = Array{Float64}(undef, S*N*N + S*N, M, M)
    #acR_mat = Array{Float64}(undef, S*N*N + S*N, M, M)
    @inbounds for m = 1:M
        acL_mat[1:S*N*N,m,m] = @view(Q_path[:,m])
        acR_mat[1:S*N*N,m,m] = @view(Q_path[:,m])
        acL_mat[S*N*N+1:end,m,m] = @view(q_path[:,m])
        acR_mat[S*N*N+1:end,m,m] = @view(q_path[:,m])
    end

    @inbounds for m = 2:M
        #A_path[:, m] .= BIn2 * A_path[:, m-1] .+ Q_path[:, m]
        @inbounds for n1 = 1:S*N*N
            @inbounds for k = 1:S*N*N
                A_path[n1, m] += BIn2[n1, k] * A_path[k, m-1]
            end
            A_path[n1, m] += Q_path[n1, m]
        end
        @inbounds for n = m-1:-1:1
            @inbounds for n1 = 1:S*N*N+S*N
                acL_mat[n1,n,m] = 0.
                acR_mat[n1,n,m] = 0.
            end
            @inbounds for n1 = 1:S*N*N+S*N
                @inbounds for k = 1:S*N*N+S*N
                    acL_mat[n1, n, m] += bL_mat[n1, k] * acL_mat[k, n, m-1]
                    acR_mat[n1, n, m] += bR_mat[n1, k] * acR_mat[k, n, m-1]
                end
            end
            @inbounds for k = 1:S*N*N
                A_path[k, m] += acL_mat[k,n,m]
                A_path[k, m] += acR_mat[k,n,m]
            end
        end
    end

    return A_path
end

function compute_terminal_regime_probs(M::Int64,msvar::MSVAR1)

    @unpack S, N, ms = msvar
    @unpack P = ms
    out_vec = [zeros(Float64, S, M) for s = 1:S]

    @inbounds for s = 1:S
        out_vec[s][s, 1] = 1.0
        @inbounds for m = 2:M
            #out_vec[s][:, m] = P * out_vec[s][:, m-1]
            @inbounds for s1 = 1:S
                @inbounds for s2 = 1:S
                    out_vec[s][s1, m] += P[s2, s1] * out_vec[s][s2, m-1]
                end
            end
        end
    end

    return out_vec
end

function compute_first_moms_mixture(Nbar::Int64, x::Vector{Float64},
            msvar::MSVAR1)

    μhat = compute_̂μ(msvar)
    BΦhat = compute_BΦ(msvar)
    a_path_tmp = compute_̄a_path(Nbar, μhat, BΦhat, repeat(x,msvar.S), msvar)
    q_path = a_path_tmp[msvar.N*msvar.S+1:end,:]
    a_path = a_path_tmp[1:msvar.N*msvar.S,:]

    return a_path, q_path
    
end

function compute_second_moms_mixture!(Q_path::Matrix{Float64},
            acL_mat::Array{Float64}, acR_mat::Array{Float64}, Nbar::Int64,
            q_path::Matrix{Float64}, msvar::MSVAR1)
    
    @unpack N, S = msvar

    Ξ = compute_Ξ(msvar)
    BΦμ = compute_BΦμ(msvar)
    μμΣΣ = compute_μμΣΣ(msvar)
    Iμ = compute_Iμ(msvar)
    μI = compute_μI(msvar)
    BΦI = compute_BΦI(msvar)
    BIΦ = compute_BIΦ(msvar)

    compute_̄Q_path!(Q_path, Nbar, q_path, zeros(S*N*N), Ξ, BΦμ, μμΣΣ, msvar)

    A_path =  compute_̄A_path!(acL_mat, acR_mat, Nbar, q_path, Q_path, msvar, Iμ,
                μI, BIΦ, BΦI)
    
    return A_path

end

function approximate_price_mixture!(Q_path::Matrix{Float64}, 
            acL_mat::Array{Float64}, acR_mat::Array{Float64}, Nbar::Int64,
            x::Vector{Float64}, δ::Vector{Float64},
            tregime_probs::Vector{Matrix{Float64}}, msvar::MSVAR1)

    @unpack S, N = msvar
    #Compute Eₜ[xₜ + ⋯ + xₜ₊ₙ₋₁]
    m1_path, q_path = compute_first_moms_mixture(Nbar, x, msvar)

    #Compute Vₜ[xₜ + ⋯ + xₜ₊ₙ₋₁]
    m2_path = compute_second_moms_mixture!(Q_path, acL_mat, acR_mat, Nbar, q_path,
                msvar)

    #For each s ∈ {1,…,S}: Compute Eₜ[sum of dicounted CFs)
    dcf_path = zeros(Float64, S, Nbar)
    tmp_m1 = 0.0
    tmp_m2 = 0.0
    @inbounds for s = 1:S
        @inbounds for n = 1:Nbar
            #dcf_path[s, n] = 
            #    exp( δ' * m1_path[1+(s-1)*N:s*N,n] + 0.5 * δ' * 
            #        ( reshape(m2_path[1+(s-1)*N*N:s*N*N,n], 2, 2) - 
            #        m1_path[1+(s-1)*N:s*N,n] * m1_path[1+(s-1)*N:s*N,n]' ) * δ )
            tmp_m1 = 0.0 #δ' * m1_path[1+(s-1)*N:s*N,n]
            @inbounds for n1 = 1:N
                tmp_m1 += δ[n1] * m1_path[n1+(s-1)*N,n]
            end

            tmp_m2 = 0.0
            tmp_ndx = 1
            @inbounds for n1 = 1:N
                tmp_m2 += δ[n1]^2 * (m2_path[tmp_ndx+(s-1)*N*N,n] - 
                    m1_path[n1+(s-1)*N,n]^2)
                tmp_ndx += S + 1
            end
            
            @inbounds for n1 = 1:N
                @inbounds for n2 = N:-1:n1+1
                    tmp_ndx = (n1 - 1)*N + n2
                    tmp_m2 += 2 * δ[n1] * δ[n2] * ( 
                        m2_path[tmp_ndx+(s-1)*N*N,n] - m1_path[n1+(s-1)*N,n] * 
                        m1_path[n2+(s-1)*N,n] )
                end
            end

            dcf_path[s, n] = exp(tmp_m1 + 0.5 * tmp_m2)
        end
    end

    η_path = zeros(Float64, S, Nbar)
    @inbounds for s = 1:S
        η_path[s, :] = sum(dcf_path .* tregime_probs[s], dims = 1)
    end

    return sum(η_path, dims = 2), η_path
end

function approximate_price_gaussian!(Q_path::Matrix{Float64}, 
    acL_mat::Array{Float64}, acR_mat::Array{Float64}, Nbar::Int64,
    x::Vector{Float64}, δ::Vector{Float64},
    tregime_probs::Vector{Matrix{Float64}}, msvar::MSVAR1)

    @unpack S, N = msvar
    #Compute Eₜ[xₜ + ⋯ + xₜ₊ₙ₋₁]
    m1_path, q_path = compute_first_moms_mixture(Nbar, x, msvar)

    #Compute Vₜ[xₜ + ⋯ + xₜ₊ₙ₋₁]
    m2_path = compute_second_moms_mixture!(Q_path, acL_mat, acR_mat, Nbar, q_path,
            msvar)

    m1_tmp = zeros(Float64, N, S, Nbar)
    m2_tmp = zeros(Float64, N, N, S, Nbar)
    for n = 1:Nbar
        m1_tmp[:,:,n] = reshape(m1_path[:,n], N, S)
        m2_tmp[:,:,:,n] = reshape(m2_path[:,n], N, N, S)
    end

    m1 = zeros(Float64, N, Nbar, S)
    m2 = zeros(Float64, N, N, Nbar, S)
    for s0 = 1:S
        for n = 1:Nbar
            for s2 = 1:S
                m1[:,n,s0] += m1_tmp[:,s2,n] * tregime_probs[s0][s2,n]
                m2[:, :, n, s0] += m2_tmp[:, :, s2, n] * tregime_probs[s0][s2,n]
            end
        end
    end


    η_path = zeros(Float64, S, Nbar)
    for s = 1:S
        for n = 1:Nbar
            η_path[s,n] = 
                exp(δ' * m1[:,n,s] + 0.5 * δ' * 
                    ( m2[:,:,n,s] - m1[:,n,s] * m1[:,n,s]' ) * δ)
        end
    end

    Q = sum(η_path, dims = 2)

    return Q, η_path, m1, m2
    
end

export compute_Ex_path
export compute_EΨ_path
export compute_Vx_path
export compute_Φ̃_path
export compute_covmat_Ψ
export compute_Φ̃_path_cov
export compute_VΨ
export compute_Eη_path
export compute_halfmc_price
export compute_mc_price
export compute_mc_price_lse
export compute_halfmc_price_lse
export compute_̂μ
export compute_BΦ
export compute_̄q_path
export compute_̄a_path
export compute_Ξ
export compute_BΦμ
export compute_μμΣΣ
export compute_̄Q_path!
export compute_μI
export compute_Iμ
export compute_BIΦ
export compute_BΦI
export compute_̄A_path
export compute_̄A_path!
export compute_terminal_regime_probs
export compute_first_moms_mixture
export compute_second_moms_mixture!
export approximate_price_mixture!
export approximate_price_gaussian!

end