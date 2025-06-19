# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Mean fields for Ising-like models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

include("discrete-maps.jl")

import Distributions: Dirichlet
import Roots: find_zero, Brent
import LinearAlgebra: norm



# Combined Ising model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function combined_fxp_currents(x::Real, 
    J::Real, θ::Real, β::Real, I::Real, R::Int, C::Vector
    ) :: Vector{Float64}
    Q = length(C)
    currs = zeros(Q)
    # some of these are simpler than the integrator currents
    # currs[1] = β*(C[1]*(J*x+I)-θ)
    currs[1:min(R+1,Q)] = [β*(sum(C[1:τmax])*(J*x+I)-θ) for τmax in 1:min(R+1,Q)]
    if Q > R+1
        for τmax in R+2:Q
            net_sum = sum([C[i]*(prod(0.5.-0.5.*tanh.(currs[1:i-1]))*x) 
                for i in 1:τmax])
            ext_sum = I*sum([C[i] for i in 1:τmax])
            currs[τmax] = β*(J*net_sum+ext_sum-θ)
        end
    end
    return currs
end

function combined_fxp_transcendental(x::Real, 
        J::Real, θ::Real, β::Real, I::Real, 
        R::Int, C::Vector) :: Real
    Q = length(C)
    currs = combined_fxp_currents(x, J, θ, β, I, R, C)
    ps = 0.5 .- 0.5*tanh.(currs)
    return (x*(R+1) + x*sum([prod(ps[1:τ-1]) for τ in 2:Q]) 
        +  x*prod(ps[1:Q])*exp(-2*currs[Q]) - 1)
end

#TODO implement this
function combined_fxp(J::Real, θ::Real, β::Real, I::Real, 
        R::Int, C::Vector) :: Vector{Float64}
    Q = length(C)
    _transcendent(x) = combined_fxp_transcendental(x, J, θ, β, I, R, C)
    sol = find_zero(_transcendent, (0, 1/(R+1)), Brent(), maxevals=200)
    currs = combined_fxp_currents(sol, J, θ, β, I, R, C)
    ps = 0.5 .- 0.5*tanh.(currs)
    return [# check this
        fill(sol, R+1)...,
        [sol*prod(ps[1:τ]) for τ in 1:Q-1]...,
        sol*prod(ps[1:Q])*exp(-2*currs[Q])
    ]
end

function combined_map_currents(x,p)
    return [(p[:C][1:τ]'*(p[:J]*x[1:τ] .+ p[:I])) for τ in 1:p[:Q]]
end

function combined_map_phase(x, p)
    if p[:θ] > 0
        phases = clamp.(combined_map_currents(x, p), 0, p[:θ])/p[:θ].*(2*pi)
        return [zeros(p[:R])..., phases..., phases[p[:Q]]]
    elseif p[:θ] < 0
        phases = clamp.(combined_map_currents(x, p), p[:θ], 0)/p[:θ].*(2*pi)
        return [zeros(p[:R])..., phases..., phases[p[:Q]]]
    else
        return zeros(p[:R]+p[:Q]+1)
    end
end

function combined_map(x, p)
    activ = tanh.(p[:β].*(combined_map_currents(x, p).-p[:θ]))/2
    R, Q = p[:R], p[:Q]
    return [
        x[R+1:R+Q]'*(0.5.+activ)+x[R+Q+1]'*(0.5.+activ[Q])  # a0
        x[1:R]                                              # a1 to a{R+1}
        x[R+1:R+Q-1].*(0.5.-activ[1:end-1])                 # a{R+2} to a{R+Q}
        (x[R+Q]+x[R+Q+1])*(0.5.-activ[Q])                   # a{R+Q+1}
    ]
end

function CombinedIMF(x0::Vector, J::Real, θ::Real, β::Real, 
        I::Real, C::Vector) :: DiscreteMap
    # deduce Q and R from x0 and C
    R = length(x0) - length(C) - 1
    Q = length(C)
    # create parameter
    p = Dict(:J => J, :θ => θ, :β => β, :I => I,
        :R=> R, :Q => Q, :C => C)
    # create featurizer
    # TODO: optimize featurizer calculations
    ft = (x, p, pc) -> [
            x[1],               # firing neurons
            sum(x[2:p[:R]]),    # refractory neurons
            sum(x[p[:R]+1:end]),# ready to fire neurons
            norm(x - pc[:fxp]), # distance to fixed point
            # kuramoto coherence
            abs(x'*exp.(im.*combined_map_phase(x, p))),  
            # kuramoto phase 
            angle(x'*exp.(im.*combined_map_phase(x, p)))
            ]
    nft = 6
    # precompute stuff
    pc = Dict(:fxp => combined_fxp(J, θ, β, I, R, C))
    return DiscreteMap{R+Q+1}("Combined IMF Model",
        combined_map, x0, x0, p, ft, nft, pc)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Initial conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function quiet_ic_mf(R::Int, Q::Int) :: Vector
    @assert R >= 0 && Q >= 0 "R and Q must be positive"
    if R == 0 && Q == 0
        return [0]
    elseif R == 0 && Q > 0
        return [zeros(Q)...,1]
    elseif R > 0 && Q == 0
        return [zeros(R)...,1]
    else
        return [zeros(R+Q)...,1]
    end
end

function spike_ic_mf(R::Int, Q::Int) :: Vector
    @assert R >= 0 && Q >= 0 "R and Q must be positive"
    if R == 0 && Q == 0
        return [1]
    elseif R == 0 && Q > 0
        return [1, zeros(Q)...]
    elseif R > 0 && Q == 0
        return [1, zeros(R)...]
    else
        return [1, zeros(R+Q)...]
    end
end

function random_ic_mf(R::Int, Q::Int) :: Vector
    @assert R >= 0 && Q >= 0 "R and Q must be positive"
    if R == 0 && Q == 0
        return [rand()]
    elseif R == 0 && Q > 0
        return rand(Dirichlet(Q+1, 1))
    elseif R > 0 && Q == 0
        return rand(Dirichlet(R+1, 1))
    else
        return rand(Dirichlet(R+Q+1, 1))
    end
end