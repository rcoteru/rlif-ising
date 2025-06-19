
include("../discrete-maps.jl")

import Roots: find_zero, Brent
import LinearAlgebra: norm

# Integrator Ising model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function integrator_fxp_currents(x::Real, 
    J::Real, θ::Real, β::Real, I::Real, C::Vector
    ) :: Vector{Float64}
    Q = length(C)
    currs = zeros(Q)
    currs[1] = β*(C[1]*(J*x+I)-θ)
    for τmax in 2:Q
        net_sum = sum([C[i]*(prod(0.5.-0.5.*tanh.(currs[1:i-1]))*x) 
            for i in 1:τmax])
        ext_sum = I*sum([C[i] for i in 1:τmax])
        currs[τmax] = β*(J*net_sum+ext_sum-θ)
    end
    return currs
end

function integrator_fxp_probs(x::Real, 
        J::Real, θ::Real, β::Real, I::Real, C::Vector
        ) :: Matrix{Float64}
    Q = length(C)
    probs = zeros(2, Q)
    activ = tanh(β*(C[1]*(J*x+I)-θ))/2
    probs[1,1], probs[2,1] = 0.5 + activ, 0.5 - activ
    for τmax in 2:Q
        net_sum = sum([C[i]*(prod(probs[2,1:i-1])*x) for i in 1:τmax])
        ext_sum = I*sum([C[i] for i in 1:τmax])
        activ = tanh(β*(J*net_sum+ext_sum-θ))/2
        probs[1,τmax], probs[2,τmax] = 0.5 + activ, 0.5 - activ
    end 
    return probs
end

function integrator_fxp_transcendental(x, J, θ, β, I, C) :: Real
    Q = length(C)
    currs = integrator_fxp_currents(x, J, θ, β, I, C)
    ps = 0.5 .- 0.5*tanh.(currs)
    return (x + x*sum([prod(ps[1:τ-1]) for τ in 2:Q]) 
        +  x*prod(ps[1:Q])*exp(-2*currs[Q]) - 1)
end

function integrator_fxp(J, θ, β, I, C) :: Vector{Float64}
    Q = length(C)
    _transcendent(x) = integrator_fxp_transcendental(x, J, θ, β, I, C)
    sol = find_zero(_transcendent, (0, 1), Brent(), maxevals=200)  
    currs = integrator_fxp_currents(sol, J, θ, β, I, C)
    ps = 0.5 .- 0.5*tanh.(currs)
    fxp = [
        sol,
        [sol*prod(ps[1:τ]) for τ in 1:Q-1]...,
        sol*prod(ps[1:Q])*exp(-2*currs[Q])
    ]
    return fxp
end

function integrator_map_currents(x, p)
    return [p[:C][1:τ]'*(p[:J]*x[1:τ] .+ p[:I]) for τ in 1:p[:Q]]
end

function combined_map_phase(x, p)
    if p[:θ] > 0
        phases = clamp.(combined_map_currents(x, p), 0, p[:θ])/p[:θ].*(2*pi)
        return [phases..., phases[p[:Q]]]
    elseif p[:θ] < 0
        phases = clamp.(combined_map_currents(x, p), p[:θ], 0)/p[:θ].*(2*pi)
        return [phases..., phases[p[:Q]]]
    else
        return zeros(p[:Q]+1)
    end
end

function integrator_map(x, p, pc)
    # activ = [tanh(p[:β]*(p[:C][1:τ]'*(p[:J]*x[1:τ] .+ p[:I])  
    #      - p[:θ]))/2 for τ in 1:p[:Q]]
    activ = tanh.(p[:β]*(integrator_map_currents(x, p).-p[:θ]))./2
    return [
        x[1:p[:Q]]'*(0.5.+activ) + x[p[:Q]+1]'*(0.5.+activ[p[:Q]]),
        x[1:p[:Q]-1].*(0.5.-activ[1:p[:Q]-1])...,
        (x[p[:Q]] + x[p[:Q]+1])*(0.5.-activ[p[:Q]])
    ]
end

function IntegratorIMF(x0::Vector, J::Real, θ::Real, β::Real, 
        I::Real, C::Vector) :: DiscreteMap
    # deduce Q from C
    Q = length(C)
    # create parameter
    p = Dict(:J => J, :θ => θ, :β => β, :Q => Q, :C => C, :I => I)
    # create featurizer
    ft = (x, p, pc) -> [
            x[1],               # firing neurons
            sum(x[1:p[:Q]]),    # unsaturated neurons
            sum(x[p[:Q]+1]),    # saturated neurons
            norm(x - pc[:fxp]), # distance to fixed point
            abs(x'*pc[:ang]),   # kuramoto coherence
            angle(x'*pc[:ang])  # kuramoto phase
            ]
    nft = 6
    # precompute stuff
    pc = Dict(
        :fxp => integrator_fxp(J, θ, β, I, C),
        :ang => exp.(im.*range(0, stop=2*pi, length=Q+1))
    )
    return DiscreteMap{Q+1}("Integrator IMF Model",
        integrator_map, x0, x0, p, ft, nft, pc)
end


function integrator_fdist_traj!(s::DiscreteMap, meas_stp::Int)
    traj_size = meas_stp + 2*s.p[:Q]
    tf = trajectory!(s, traj_size, ft=false)
    Q, scan = s.p[:Q], s.p[:Q]
    fdist = zeros(Float64, (meas_stp, 2, Q+1))
    for i in Q+1:traj_size-scan
        fdist[i-scan,1,:] = tf[i,:]
        #TODO this is not correct, find a way to do this from the trajectory
        # although most of the time it should be correct enough
        for τ in 1:Q+1
            fdist[i-scan,2,τ] = min(tf[i+τ-1,1], 
                (1-sum(fdist[i-scan,2,:])))
        end
    end
    return fdist
end

function integrator_entropy!(s::DiscreteMap, meas_stp::Int)
    fdist = integrator_fdist_traj!(s, meas_stp+2)
    S, Q = zeros(meas_stp,2), s.p[:Q]
    for i in 1:meas_stp
        # p(n) / p(\hat{n})
        Nf = fdist[i,1,:]
        Nb = fdist[i+2,2,:]
        # local fields 
        hf = s.p[:β].*(integrator_map_currents(Nf, s.p).-s.p[:θ])
        hb = s.p[:β].*(integrator_map_currents(Nb, s.p).-s.p[:θ])
        hf = [hf[1:Q]..., hf[Q]]
        hb = [hb[1:Q]..., hb[Q]]

        S[i,1] = Nf'*(-hf.*tanh.(hf) + log.(2*cosh.(hf)))

        hf2 = hf*ones(1,Q+1)
        hb2 = ones(Q+1,1)*hb'
        Sr2 = (-hb2.*tanh.(hf2) + log.(2*cosh.(hb2)))

        S[i,2] = Nf'*Sr2*Nb
    end
    return S
end