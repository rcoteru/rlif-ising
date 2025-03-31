# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Mean fields for Ising-like models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

include("discrete-maps.jl")

import Distributions: Dirichlet
import Roots: find_zero, Brent
import LinearAlgebra: norm

# Vanilla Ising model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function vanilla_map(x, p)
    return [0.5 + 0.5*tanh(p[:β]*(p[:J]*x[1]+p[:I]-p[:θ]))]   
end

"""
VanillaIMF(x0, J, θ, β)

Returns a DiscreteMap object of the regular Ising model.

# Arguments
- `x0::Real`: initial condition
- `J::Real`: synaptic coupling
- `θ::Real`: external field
- `β::Real`: inverse temperature

# Returns
- `DiscreteMap`: object representing the regular Ising model
"""
function VanillaIMF(x0::Vector, J::Real, θ::Real, β::Real, I::Real) :: DiscreteMap
    # create parameters
    p = Dict(:J => J, :θ => θ, :β => β, :I => I)
    # create featurizer
    ft = (x, p, pc) -> x
    nft = 1
    # precompute stuff
    pc = Dict()
    return DiscreteMap{1}("Vanilla Ising",
        vanilla_map, x0, x0, p, ft, nft, pc)
end

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
    S = zeros(meas_stp,2)
    for i in 1:meas_stp
        # p(n) / p(\hat{n})
        Nf = fdist[i,1,:]
        Nb = fdist[i+2,2,:]
        # local fields 
        hf = s.p[:β]*(s.p[:J]*Nf[1] - s.p[:θ])
        hb = s.p[:β]*(s.p[:J]*Nb[1] - s.p[:θ])
        # entropy
        S[i,1] = Nf[R+1]*(-hf*tanh(hf) + log(2*cosh(hf)))
        S[i,2] = Nb[R+1]*(-hb*tanh(hb) + log(2*cosh(hb)))
    end
    return S
end

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
    #p = integrator_fxp_probs(x, J, θ, β, I, C)
    # return (1/x - 1 - sum([prod(p[2,1:τ-1]) for τ in 2:Q]) 
    #    - p[2,Q]/p[1,Q]*prod(p[2,1:Q]))
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

function integrator_map(x, p)
    activ = [tanh(p[:β]*(p[:C][1:τ]'*(p[:J]*x[1:τ] .+ p[:I])  
        - p[:θ]))/2 for τ in 1:p[:Q]]
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

# Combined Ising model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# """
# complete_ising_fxp(J, θ, β, R, Q, C)

# Returns the analystical fixed point of the complete Ising model.

# # Arguments
# - `J::Real`: synaptic coupling
# - `θ::Real`: external field
# - `β::Real`: inverse temperature
# - `R::Int`: refractory period
# - `Q::Int`: memory of the system
# - `C::Vector`: memory weights

# # Returns
# - `Vector{Float64}`: fixed point of the complete Ising model
# """
# function complete_ising_fxp(J, θ, β, R, Q, C) :: Vector{Float64}
#     _probs(x) = 0.5 .+ tanh.(β.*(J.*x.*cumsum(C).+θ))./2
#     function _transcendent(x)
#         pp = _probs(x)
#         pn = 1 .- pp
#         return 1/x - (R+1) - sum([prod(pn[1:τ-1]) for τ in 2:Q-1]) - 1/pp[end]*prod(pn[1:end-1])
#     end
#     sol = find_zero(_transcendent, (0, 1/(R+1)), Brent(), maxevals=200)
#     pp = _probs(sol)
#     pn = 1 .- pp
#     return [
#         fill(sol, R+1)
#         [sol*prod(pn[1:τ]) for τ in 1:Q-2]
#         sol*prod(pn[1:Q-1])/pp[end]
#         ]
# end

function combined_fxp_currents(x::Real, 
    J::Real, θ::Real, β::Real, I::Real, C::Vector
    ) :: Vector{Float64}
    Q = length(C)
    currs = zeros(Q)
    # some of these are simpler than the integrator currents
    # currs[1] = β*(C[1]*(J*x+I)-θ)
    currs[1:R+1] = [β*(C[1:r]*(J*x+I)-θ) for r in 1:R+1]
    for τmax in R+2:Q
        net_sum = sum([C[i]*(prod(0.5.-0.5.*tanh.(currs[1:i-1]))*x) 
            for i in 1:τmax])
        ext_sum = I*sum([C[i] for i in 1:τmax])
        currs[τmax] = β*(J*net_sum+ext_sum-θ)
    end
    return currs
end

function combined_fxp_transcendental(x::Real, 
        J::Real, θ::Real, β::Real, I::Real, 
        R::Int, C::Vector) :: Real
    Q = length(C)
    currs = combined_fxp_currents(x, J, θ, β, I, C)
    ps = 0.5 .- 0.5*tanh.(currs)
    return (x + x*sum([prod(ps[1:τ-1]) for τ in 2:Q]) 
        +  x*prod(ps[1:Q])*exp(-2*currs[Q]) - 1)
end

#TODO implement this
function combined_fxp(J::Real, θ::Real, β::Real, I::Real, 
        R::Int, C::Vector) :: Vector{Float64}
    Q = length(C)
    _transcendent(x) = combined_fxp_transcendental(x, J, θ, β, I, R, C)
    sol = find_zero(_transcendent, (0, 1/(R+1)), Brent(), maxevals=200)
    currs = combined_fxp_currents(sol, J, θ, β, I, C)
    ps = 0.5 .- 0.5*tanh.(currs)
    return [# check this
        fill(sol, R+1),
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
        :R=> R, :Q => Q, :C => C, )
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
    pc = Dict(
        :fxp => zeros(R+Q+1),#combined_fxp(J, θ, β, I, R, C))
        :ang => exp.(im.*range(0, stop=2*pi, length=Q+R+1))
    )
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