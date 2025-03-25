include("_dm-common.jl")

using LinearAlgebra: diagm, I
export DiscreteHopfieldMF, m_overlap, A_classic, A_cycle

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function _hopfield(x0, ξ, A, θ, β) :: DiscreteMap
    M, N = size(ξ)
    # check all values in x0 are within [-1, 1]
    @assert all(-1 .<= x0 .<= 1) "IC x0 must be within [-1, 1]"
    # check x0 is a vector of length M
    @assert size(x0) == (M,) "IC x0 must be a vector of length M"
    # update function
    function _hopfield_map(x, p)
        return p[:ξ]*tanh.(p[:B]*(p[:ξ]'p[:A]*x .+ p[:H]))/p[:N]
    end
    # featurizer
    ft = (x, p, pc) -> x
    # precompute stuff
    pc = Dict()
    p = Dict(:N => N, :M => M, :ξ => ξ, :A => A, :H => θ, :B => β)
    return DiscreteMap{M}("Hopfield Model",
        _hopfield_map, x0, x0, p, ft, pc)
end

function _active_hopfield(x0, ξ, A, θ, β) :: DiscreteMap
    M, N = size(ξ)
    # check all values in x0 are within [-1, 1]
    @assert all(-1 .<= x0 .<= 1) "IC x0 must be within [-1, 1]"
    # check x0 is a vector of length M
    @assert size(x0) == (M,) "IC x0 must be a vector of length M"
    # update function
    function _active_hopfield_map(x, p)
        return 0.5*(ξ*ones(N)/p[:N] + p[:ξ]*tanh.(p[:B]*(p[:ξ]'p[:A]*x .+ p[:H]))/p[:N])
    end
    # featurizer
    ft = (x, p, pc) -> μ2m(x, pc[:ξavg])
    # precompute stuff
    pc = Dict(:ξavg => ξ*ones(N)/N)
    p = Dict(:N => N, :M => M, :ξ => ξ, :A => A, :H => θ, :B => β)
    return DiscreteMap{M}("Active Hopfield Model",
        _active_hopfield_map, x0, x0, p, ft, pc)
end

function _refrac_active_hopfield(x0, ξ, A, θ, β, R) :: DiscreteMap
    M, N = size(ξ)
    # check all values in x0 are within [-1, 1]
    @assert all(-1 .<= x0 .<= 1) "IC x0 must be within [-1, 1]"
    # check x0 is a vector of length M
    @assert size(x0) == (M,) "IC x0 must be a vector of length M"
    # update function
    function refrac_active_hopfield_map(x, p)
        # common vars
        a = tanh.(p[:B]*(p[:ξ]'p[:A]*x[1:p[:M]] .+ p[:H]))    
        b = x[p[:M]+p[:R]]
        firing = ones(p[:N])'*a/p[:N]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        mu = b*0.5*(p[:ξ]*ones(N) + p[:ξ]*a)/p[:N]  # μ_{t+1}    
        N0 = b*0.5*(1 + firing)                     # N_{0,t+1} 
        N1R_1 = x[p[:M]+1:end-2]                    # N_{1:R-1,t+1}^R
        NR = x[end-1] + b*0.5*(1 - firing)          # N_{R,t+1}^R
        return [mu..., N0, N1R_1..., NR]
    end
    p = Dict(:N => N, :M => M, :ξ => ξ, :A => A, :H => θ, :B => β, :R => R)
    return DiscreteMap{M}("Refractory Active Hopfield Model",
        _hopfield_map, [x0], [x0], p)
end

function DiscreteHopfieldMF(x0, 
        ξ :: Matrix{<:Int}, A :: Matrix{<:Real}, 
        θ :: Real, β :: Real, R :: Int = 0,
        active::Bool =  false, refrac::Bool = false
    ) :: DiscreteMap
    M, N = size(ξ)
    # check x0 is a vector of length M
    @assert size(x0) == (M,) "IC x0 must be a vector of length M"
    # check A is a square matrix of size M
    @assert size(A) == (M, M) "Parameter A must be a square matrix of size M"
    # check β is positive
    @assert β >= 0 "Parameter β must be positive"
    if active && refrac
        return _refrac_active_hopfield(x0, ξ, A, θ, β, R)
    elseif active
        return _active_hopfield(x0, ξ, A, θ, β)
    elseif refrac
        throw(ArgumentError("Refractory period not implemented for Hopfield model"))
    else
        return _hopfield(x0, ξ, A, θ, β)
    end
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Initial conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

m_overlap(S::Vector, ξ::Matrix) :: Vector =  ξ*S./size(ξ,2)
μ_overlap(S::Vector, ξ::Matrix) :: Vector = 0.5*(ξ*S./size(ξ,2) + ξ*ones(N)/size(ξ,2))

m2μ(m::Vector, ξ::Matrix) :: Vector = 0.5*(ξ*m + ξ*ones(N)/size(ξ,2))
μ2m(μ::Vector, ξ::Matrix) :: Vector = 2*μ - ξ*ones(N)/size(ξ,2)

m2μ(m::Vector, ξavg::Vector) :: Vector = 0.5*(ξ*m + ξavg)
μ2m(μ::Vector, ξavg::Vector) :: Vector = 2*μ - ξavg


A_classic(N:: Integer) :: Matrix = Matrix(I(N))
function A_cycle(N:: Integer) :: Matrix 
    @assert N > 2 "N must be greater than 2"
    return diagm(N, N, -1 => -ones(N-1), 0 => ones(N), 
            1 => ones(N-1), N-1 => [-1], -N+1 => [1])
end