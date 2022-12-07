using MixedLognormalPricing
using Random
using LinearAlgebra
using Parameters
using BenchmarkTools

seed_num = 1234
rng = Xoshiro(seed_num)

Π = [0.9 0.1 ; 0.5 0.5]
μ₁ = [0.0, 0.0]
μ₂ = [0.0 + randn(rng), 0.0 + randn(rng)]
μ = [μ₁, μ₂]

Φ₁ = [0.9 0.0 ; 0.0 0.9]
Φ₂ = [0.1 + randn(rng) 0.0 ; 0.0 0.1 + randn(rng)]
Φ = [Φ₁, Φ₂]

Σ₁ = [0.1 0.0 ; 0.0 0.1]
Σ₂ = [0.1 + rand(rng) 0.0 ; 0.0 0.1 + rand(rng)]
Ω = [Σ₁, Σ₂]

ms = MS(S = 2, P = Π)
msvar = MSVAR1(ms = ms, N = 2, μ = μ, Φ = Φ, Ω = Ω)


function construct_Ω(msvar::MSVAR1)
    # Ω = bdiag(A₁,\ldots,Aₙ) × (H ⊗ Iₙ)
    @unpack Φ, N, ms = msvar
    @unpack P = ms
    return BlockDiagonal(Φ) * kron(P', Matrix(I, N, N))
end

Ω = construct_Ω(msvar)
@code_warntype construct_Ω(msvar)
@btime construct_Ω(msvar)

function construct_Ω2(msvar::MSVAR1)
    # Ω = bdiag(A₁,\ldots,Aₙ) × (H ⊗ Iₙ)
    @unpack Φ, N, ms = msvar
    @unpack P, S = ms
    bdiagΦ = zeros(N*S, N*S)
    for i in 1:S
            bdiagΦ[(i-1)*N+1:i*N, (i-1)*N+1:i*N] = Φ[i]
    end
    return bdiagΦ * kron(P', Matrix(I, N, N))
end

@btime construct_Ω2(msvar)

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

C = construct_C(msvar)
function construct_Ωtil(msvar::MSVAR1)

    @unpack ms, N  = msvar
    @unpack S, P = ms
    H = P'

    NM = S * N

    Ω = construct_Ω(msvar)
    C = construct_C(msvar)

    Ωtil = zeros(Float64, NM + S, NM + S)
    Ωtil[1:NM, 1:NM] .= Ω
    Ωtil[1:NM, NM+1:end] .= C * H
    Ωtil[NM+1:end, NM+1:end] .= H

    return Ωtil

end

Ωtil = construct_Ωtil(msvar)

@btime construct_Ωtil(Ω, C, msvar)

function construct_AAhat(msvar::MSVAR1)
    @unpack Φ, N, ms = msvar
    @unpack S = ms

    AAhat = zeros(Float64, N * N * S, N * N * S)
    for s = 1:S
        AAhat[(s-1)*N*N+1:s*N*N, (s-1)*N*N+1:s*N*N] .= kron(Φ[s],Φ[s])
    end
    
    return AAhat
end

 AAhat = construct_AAhat(msvar)

 function construct_Ξ(msvar::MSVAR1)
    # Ξ = bdiag(A₁ ⊗ A₁,…,Aₘ ⊗ Aₘ) * (H ⊗ Iₙₙ)
    @unpack ms, N  = msvar
    @unpack S, P = ms
    H = P'

    AAhat = construct_AAhat(msvar)

    return AAhat * kron(H, Matrix(I, N * N, N * N))
end

Ξ = construct_Ξ(msvar)

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

@btime VVhat = construct_VVhat(msvar)

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

cchat = construct_cchat(msvar)
@btime cchat = construct_cchat(msvar)


function construct_Vchat(msvar::MSVAR1)
    # Vchat = [hatVV + hatcc]
    VVhat = construct_VVhat(msvar)
    cchat = construct_cchat(msvar)

    return VVhat + cchat
end

@code_warntype construct_Vchat(msvar)

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

DAChat = compute_DAChat(msvar)

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

Ξtil = compute_Ξtilde(msvar)

@btime compute_Ξtilde(msvar)

A = cumprod([Ξtil for i=1:100])

spΞtil = sparse(Ξtil)

A[1]
A[3] .- A[1]*A[1]*A[1]

x0 = [0.0; 0.0]
xx0 = (x0*x0')[:]
π0 = [1.0; 0.0]

Qtil0 = [repeat(xx0,2,1);repeat(x0,2,1);π0]

Ξtil^2 * Qtil0

