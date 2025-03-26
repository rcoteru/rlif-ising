# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Spin Model w/RP&RI - Julia Implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

using DataStructures: CircularBuffer
import Random: randperm
using Statistics: mean

# Data structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

"""
IsingModel

Struct to hold the state of the system and its parameters.

# Fields
- `J::Real`: synaptic coupling
- `R::Integer`: refractory period
- `C::Vector{Real}`: memory weights (Q)
- `s::Vector{Int}`: state of the system
- `n::Vector{Int}`: timesteps since last firing
- `a::Vector{Real}`: memory vector (Q)
- `θ::Real`: external field
- `β::Real`: inverse temperature
"""
struct IsingModel
    name:: String       # name of the model
    N :: Int            # number of neurons 
    Q :: Int            # memory of the system
    # parameters of the system
    J :: Real           # synaptic coupling 
    R :: Integer        # refractory period
    C :: Vector{Real}   # memory weights (Qx1)
    # state of the system
    s :: Vector{Int}    # state of the system
    n :: Vector{Int}    # timesteps since last firing
    a :: Vector{Real}   # memory vector (Q)
    # external variables
    θ :: Real           # external field 
    β :: Real           # inverse temperature
    I :: Real           # external input
end

function Base.show(io::IO, s::IsingModel) 
    println(io, "$(s.name): N=$(s.N); J=$(s.J); θ=$(s.θ); β=$(s.β); R=$(s.R); Q=$(s.Q)")
    println(io, "p(n) -> $(n2N(s))")
end

# Initial Conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

spike_ic_sm(N::Int) :: Vector{Int} = zeros(Int, N);
random_ic_sm(N::Int, Nmax::Int) :: Vector{Int} = rand(
    range(0, Nmax, step=1), N);

# Conversions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function s2a(s::Vector)::Real
    return mean(s)/2+0.5
end

function n2s(n::Vector)::Vector
    return map(n -> n == 0 ? 1 : -1, n) 
end

function n2N(n::Vector, N_len::Int)
    N = zeros(N_len)
    for i in 1:N_len-1 N[i] = sum(n.==i-1) end
    N[N_len] = sum(n.>=N_len-1)
    return N./length(n)
end
function n2N(s::IsingModel)
    if s.R == 0 && s.Q == 1 N_len = 1
    elseif s.R == 0 && s.Q > 0 N_len = s.Q+1
    elseif s.R > 0 && s.Q == 1 N_len = s.R+1
    else N_len = s.R+s.Q+1
    end
    return n2N(s.n, N_len)
end

function n2a(n::Vector, Q::Int)
    a = zeros(Q)
    for i in 1:Q a[i] = sum(n.==i-1) end
    return a./length(n)
end
function n2a(s::IsingModel)
    a = zeros(s.Q)
    for i in 1:s.Q a[i] = sum(s.n.==i-1) end
    return a./length(s.n)  
end

# Constructors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function VanillaSM(J, θ, β, I, n)
    R = 0
    Q, C = 1, [1]
    return IsingModel("Vanilla Ising",
        length(n), Q, J, R, C, n2s(n), n, n2a(n,Q), θ, β, I)
end

function RefractiveSM(J, θ, β, I, n, R)
    Q, C = 1, [1]
    return IsingModel("Refractive Ising",
        length(n), Q, J, R, C, n2s(n), n, n2a(n,Q), θ, β, I)
end

function IntegratorSM(J, θ, β, I, n, C)
    R = 0
    return IsingModel("Integrator Ising",
        length(n), length(C), J, R, C, n2s(n), n, n2a(n,length(C)), θ, β, I)
end

function CombinedSM(J, θ, β, I, n, R, C)
    return IsingModel("RLIF Ising",
        length(n), length(C), J, R, C, n2s(n), n, n2a(n,length(C)), θ, β, I)
end

# Current/Probability Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function local_current(s::IsingModel, τ::Int)::Float64
    return s.C[1:τ]'*(s.J*s.a[1:τ].+s.I)
end
function local_currents(s::IsingModel)::Vector{Float64}
    cτ = [local_current(s, τ) for τ in 1:s.Q]
    return [cτ[min(max(1,n+1-s.R), s.Q)] for n in s.n]
end

function fprob(s::IsingModel)::Vector{Float64}
    return 0.5*(1 .+tanh.(s.β*(local_currents(s).-s.θ)))
end

function fprob(s::IsingModel, i::Int)::Float64
    τ = min(s.n[i]+1, s.Q)
    return 0.5*(1+tanh(s.β*(local_current(s, τ)-s.θ)))
end

function rcheck(s::IsingModel)::Vector{Bool}
    return s.n .>= s.R
end

# Update Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
function parallel_update!(s:: IsingModel)::Nothing
    fired = ((rand(s.N) .< fprob(s)) .& rcheck(s))
    s.s[:] .= map(f -> f ? 1 : -1, fired)
    s.n[:] .= map((f, n) -> f ? 0 : n + 1, fired, s.n)
    s.a[:] .= n2a(s)
    return nothing
end

function sna_single_update(s::IsingModel, i::Int, ns::Int) :: Nothing
    if s.s[i] == 1
        if ns == -1
            s.s[i] = -1
            s.a[1] -= 1/s.N
            if s.Q > 1
                s.a[2] += 1/s.N
            end
            s.n[i] += 1
            return nothing
        end
    elseif s.s[i] == -1
        if ns == 1
            s.s[i] = 1
            s.a[1] += 1/s.N
            if s.Q >= s.n[i]+1
                s.a[s.n[i]+1] -= 1/s.N
            end
            s.n[i] = 0
            return nothing
        elseif ns == -1
            if s.Q >= s.n[i] + 1
                s.a[s.n[i]+1] -= 1/s.N
            end
            if s.Q >= s.n[i] + 2
                s.a[s.n[i]+2] += 1/s.N
            end
            s.n[i] += 1
            return nothing
        end
    end
    return nothing
end

"""
glauber_step!(s::IsingModel, i::Int)

Update the state of the system for a single neuron.

# Arguments
- `s::IsingModel`: struct containing the state of the system
- `i::Int`: index of the neuron to update
"""
function glauber_step!(s::IsingModel, i::Int)::Nothing
    # check first if neuron is refractive to save compute
    if s.n[i] < s.R
        sna_single_update(s, i, -1)
        return nothing
    end
    ns = rand() < fprob(s, i) ? 1 : -1
    sna_single_update(s, i, ns)
    return nothing
end

"""
sequential_update!(s::IsingModel)

Update the state of the system sequentially.

# Arguments
- `s::IsingModel`: struct containing the state of the system
"""
function sequential_update!(s::IsingModel)::Nothing
    # update neurons in random order
    perm = randperm(s.N)
    for i in perm
        glauber_step!(s, i)
    end
    # update memory vector to prevent numerical errors
    s.a[:] .= n2a(s)
    return nothing
end

# Trajectory Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

"""
forward!(s::IsingModel, nsteps::Int; parallel::Bool)

Update the state of the system for a given number of steps.

# Arguments
- `s::IsingModel`: struct containing the state of the system
- `nsteps::Int`: number of steps to update the system
- `parallel::Bool`: whether to update the system in parallel
"""
function forward!(s::IsingModel, nsteps:: Int; parallel :: Bool = false)::Nothing
    if parallel
        for _ in 1:nsteps parallel_update!(s) end
    else
        for _ in 1:nsteps sequential_update!(s) end
    end
    return nothing
end

function spinwise_traj!(s::IsingModel, nsteps::Int; parallel :: Bool)
    sps, lcs = zeros(Int8, nsteps, s.N), zeros(Float64, nsteps, s.N)
    sps[1,:], lcs[1,:] = s.s[:], local_currents(s)
    if parallel
        for i in 2:nsteps
            parallel_update!(s)
            sps[i,:], lcs[i,:] = s.s[:], local_currents(s)
        end
    else
        for i in 1:nsteps
            sequential_update!(s)
            sps[i,:], lcs[i,:] = s.s[:], local_currents(s)
        end
    end
    return sps, lcs
end

function network_traj!(s::IsingModel, nsteps :: Int; parallel :: Bool)::Matrix{Float64}
    if s.R == 0 && s.Q == 1 
        fxp = vanilla_ising_fxp(s.J, s.θ, s.β)
        idxs = [1:1]
    elseif s.R == 0 && s.Q > 0 
        fxp = integrator_ising_fxp(s.J, s.θ, s.β, s.C)
        idxs = [1:1, 2:s.Q, s.Q+1:s.Q+1]
    elseif s.R > 0 && s.Q == 1 
        fxp = refractive_ising_fxp(s.J, s.θ, s.β, s.R)
        idxs = [1:1, 2:s.R, s.R+1:s.R+1]
    else 
        fxp = complete_ising_fxp(s.J, s.θ, s.β, s.R, s.C)
        idxs = [1:1, 2:s.R, s.R+1:s.R+s.Q, s.R+s.Q+1:s.R+s.Q+1]
    end
    traj = zeros(Float64, nsteps, length(idxs)+1)
    if parallel
        for i in 1:nsteps
            parallel_update!(s)
            N = n2N(s)
            for (j, idx) in enumerate(idxs)
                traj[i,j] = sum(N[idx])
            end
            traj[i,end] = norm(fxp - N)
        end
    else
        for i in 1:nsteps
            sequential_update!(s)
            N = n2N(s)
            for (j, idx) in enumerate(idxs)
                traj[i,j] = sum(N[idx])
            end
            traj[i,end] = norm(fxp - N)
        end
    end
    return traj
end

# Statistics Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function stats!(s::IsingModel, nsteps::Int; parallel::Bool) :: Vector{Float64}
    traj = network_traj!(s, nsteps, parallel=parallel)
    return [mean(traj, dims=1)[:]..., std(traj, dims=1)...]
end

# Useful Graphs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #