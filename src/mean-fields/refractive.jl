include("../discrete-maps.jl")

import Roots: find_zero, Brent
import LinearAlgebra: norm

# Refractive Ising model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function refractive_fxp_transcendental(x::Real, 
        J::Real, θ::Real, β::Real,I::Real,R::Int)
   return tanh(β*(J*x+I-θ))-2*x/(1-x*R)+1 
end

function refractive_fxp(J::Real, θ::Real, β::Real, I::Real, R::Int) :: Vector
    _transcendent(x) = refractive_fxp_transcendental(x, J, θ, β, I, R)
    sol = find_zero(_transcendent, (0, 1/R), Brent(), maxevals=100)
    return [fill(sol, R)..., 1-sol*R]
end

function RefractiveIMF(x0::Vector, J::Real, θ::Real, β::Real, I::Real) :: DiscreteMap
    # deduce R from x0
    R = length(x0) - 1
    # update function
    function _refractive_ising_map(x, p)
        activ = tanh(p[:β]*(p[:J]*x[1]+p[:I]-p[:θ]))/2
        return [
            x[p[:R]+1]*(0.5+activ), 
            x[1:p[:R]-1]..., 
            x[p[:R]] + x[p[:R]+1]*(0.5-activ)
        ]
    end
    # create parameter
    p = Dict(:J => J, :θ => θ, :β => β, :I=>I, :R => R)
    # create featurizer
    ft = (x, p, pc) -> [
            x[1],               # firing neurons
            sum(x[2:p[:R]]),    # refractory neurons
            x[p[:R]+1],         # ready to fire neurons
            norm(x - pc[:fxp]), # distance to fixed point
            abs(x'*pc[:ang]),   # kuramoto coherence
            angle(x'*pc[:ang])  # kuramoto phase
            ]
    nft = 6
    # precompute stuff
    pc = Dict(
        :fxp => refractive_fxp(J,θ,β,I,R),
        :ang => exp.(im.*range(0, stop=2*pi, length=R+1))
        )
    return DiscreteMap{R+1}("Refractive Model",
        _refractive_ising_map, x0, x0, p, ft, nft, pc)
end

function refractive_fdist_traj!(s::DiscreteMap, meas_stp::Int)
    traj_size = meas_stp + 2*s.p[:R]
    tf = trajectory!(s, traj_size, ft=false)[:,1]
    R, scan = s.p[:R], s.p[:R]
    fdist = zeros(Float64, (meas_stp, 2, R+1))
    for i in R+1:traj_size-scan
        fdist[i-scan,1,:] = reverse(tf[i-scan:i])
        fdist[i-scan,2,:] = tf[i:i+scan]
        fdist[i-scan,1,R+1] = 1 - sum(fdist[i-scan,1,1:R])
        fdist[i-scan,2,R+1] = 1 - sum(fdist[i-scan,2,1:R])
    end
    return fdist
end

function refractive_entropy!(s::DiscreteMap, meas_stp::Int)
    fdist = refractive_fdist_traj!(s, meas_stp+2)
    S, R = zeros(meas_stp,2), s.p[:R]
    for i in 1:meas_stp
        # p(n) / p(\hat{n})
        Nf = fdist[i,1,:]
        Nb = fdist[i+2,2,:]
        # local fields 
        hf = s.p[:β]*(s.p[:J]*Nf[1] - s.p[:θ])
        hb = s.p[:β]*(s.p[:J]*Nb[1] - s.p[:θ])
        # entropy
        S[i,1] = (-hf*tanh(hf) + log(2*cosh(hf)))
        S[i,2] = (-hb*tanh(hf) + log(2*cosh(hb)))
    end
    return S
end