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
function s2n(s::Vector)::Vector
    return map(s -> s == 1 ? 0 : 1, s)
end
function S2n(s::Matrix, ncap ::Int, rev::Bool=true)::Vector
    n = zeros(Int, size(s, 2))
    for i in 1:size(s, 2)
        if rev
            idx = findfirst(reverse(s[:,i].==1))
        else
            idx = findfirst(s[:,i].==1)
        end
        n[i] = (isnothing(idx) ? ncap : idx[1]-1)
    end
    return n
end
function n2s(n::Vector)::Vector
    return map(n -> n == 0 ? 1 : -1, n) 
end
function n2S(n::Vector, ncap::Int)::Matrix
    S = -1*ones(Int8, length(n), ncap)
    [S[i, n[i]+1] = 1 for i in 1:length(n)]
    return S
end

function n2a(n::Vector, Q::Int)
    a = zeros(Q)
    for i in 1:Q a[i] = sum(n.==i-1) end
    return a./length(n)
end
function N2a(N::Vector, Q::Int)
    @assert length(N) > Q
    return N[1:Q]
end
function n2N(n::Vector, N_len::Int)
    N = zeros(N_len)
    for i in 1:N_len-1 N[i] = sum(n.==i-1) end
    N[N_len] = sum(n.>=N_len-1)
    return N./length(n)
end
function S2N(S::Matrix, N_len::Int, rev::Bool=true)::Vector
    n2N(S2n(S, N_len, rev), N_len)
end

# Metrics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

function Ncap(s::IsingModel) :: Integer
    if s.R == 0 && s.Q == 1 N_len = 1
    elseif s.R == 0 && s.Q > 0 N_len = s.Q+1
    elseif s.R > 0 && s.Q == 1 N_len = s.R+1
    else N_len = s.R+s.Q+1
    end
    return N_len
end
function n2a(s::IsingModel)
    a = zeros(s.Q)
    for i in 1:s.Q a[i] = sum(s.n.==i-1) end
    return a./length(s.n)  
end
function n2N(s::IsingModel)
    return n2N(s.n, Ncap(s))
end
function n2img(s::IsingModel, clamp::Bool=true)
    @assert isinteger(sqrt(length(s.n))) "n2img: n is not a square"
    side = Int(sqrt(length(s.n)))
    img = reshape(s.n, (side, side))
    if clamp
        return clamp.(img, 0, Ncap(s))./Ncap(s)  
    else
        return img
    end
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

function local_current(J::Real, τ::Int, 
    a::Vector, C::Vector, I::Real)::Float64 
    return C[1:τ]'*(J*a[1:τ].+I)
end
function local_current(s::IsingModel, τ::Int)::Float64
    #return s.C[1:τ]'*(s.J*s.a[1:τ].+s.I)
    return local_current(s.J, τ, s.a, s.C, s.I)
end
function local_currents(s::IsingModel)::Vector{Float64}
    cτ = [local_current(s, τ) for τ in 1:s.Q]
    return [cτ[min(max(1,n+1-s.R), s.Q)] for n in s.n]
end

function kuramoto_phases(s::IsingModel)::Vector{Float64}
    if s.θ > 0
        currs = [local_current(s, τ) for τ in 1:s.Q]
        phases = clamp.(currs, 0, s.θ)/s.θ.*(2*pi)
        return [zeros(s.R)..., phases..., phases[s.Q]]
    elseif s.θ < 0
        currs = [local_current(s, τ) for τ in 1:s.Q]
        phases = clamp.(currs, s.θ, 0)/s.θ.*(2*pi)
        return [zeros(s.R)..., phases..., phases[s.Q]]
    else
        return zeros(s.R+s.Q+1)
    end    
end

function kuramoto(s::IsingModel)::Complex
    phases = kuramoto_phases(s)
    probs  = n2N(s)
    return probs'*exp.(im.*phases)
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

function spinwise_traj!(s::IsingModel, nsteps::Int; parallel :: Bool = true)
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

function fdist_traj!(sm::IsingModel, nsteps::Int; parallel :: Bool = true)
    # get trajectory
    scan = Ncap(sm)-1
    traj_size = nsteps + 2*scan
    sps, _ = spinwise_traj!(sm, traj_size, parallel=parallel)
    # convert to p(n) and p(\hat{n})
    fdist = zeros(nsteps, 2, Ncap(sm))
    for i in scan+1:nsteps+scan
        fdist[i-scan, 1, :] = S2N(sps[i-scan:i,:], Ncap(sm), true)
        fdist[i-scan, 2, :] = S2N(sps[i:i+scan,:], Ncap(sm), false)
    end
    return fdist
end

function network_traj!(s::IsingModel, nsteps :: Int; parallel :: Bool)::Matrix{Float64}
    if s.R == 0 && s.Q == 1 
        fxp = vanilla_ising_fxp(s.J, s.θ, s.β, s.I)
        idxs = [1:1]
    elseif s.R == 0 && s.Q > 0 
        fxp = integrator_fxp(s.J, s.θ, s.β, s.I, s.C)
        ang = exp.(im.*range(0, stop=2*pi, length=Q+1))
        idxs = [1:1, 2:s.Q, s.Q+1:s.Q+1]
    elseif s.R > 0 && s.Q == 1 
        fxp = refractive_fxp(s.J, s.θ, s.β, s.I, s.R)
        ang = exp.(im.*range(0, stop=2*pi, length=R+1))
        idxs = [1:1, 2:s.R, s.R+1:s.R+1]
    else 
        fxp = complete_fxp(s.J, s.θ, s.β, s.I, s.R, s.C)
        ang = exp.(im.*range(0, stop=2*pi, length=R+Q+1))
        idxs = [1:1, 2:s.R, s.R+1:s.R+s.Q, s.R+s.Q+1:s.R+s.Q+1]
    end
    traj = zeros(Float64, nsteps, length(idxs)+2)
    if parallel
        for i in 1:nsteps
            parallel_update!(s)
            N = n2N(s)
            for (j, idx) in enumerate(idxs)
                traj[i,j] = sum(N[idx])
            end
            traj[i,end-1] = norm(fxp - N)
            traj[i,end] = abs(N'*ang)
        end
    else
        for i in 1:nsteps
            sequential_update!(s)
            N = n2N(s)
            for (j, idx) in enumerate(idxs)
                traj[i,j] = sum(N[idx])
            end
            traj[i,end-1] = norm(fxp - N)
            traj[i,end] = abs(N'*ang)
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

function entropy!(sm::IsingModel, meas_stp::Int, parallel :: Bool = true)
    #get trajectory
    fdist = fdist_traj!(sm, meas_stp+2, parallel=parallel)
    # calculate entropy
    S = zeros(meas_stp, 2)
    for i in 1:meas_stp
        Nf = fdist[i, 1, :]
        Nb = fdist[i+2, 2, :] 

        hf = [sm.β*(local_current(sm.J,τ,N2a(Nf,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]
        hb = [sm.β*(local_current(sm.J,τ,N2a(Nb,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]
        if sm.Q>1
            hf = [hf[1:sm.Q]..., hf[sm.Q]]
            hb = [hb[1:sm.Q]..., hb[sm.Q]]
        end
        
        if sm.Q>1
            NfR = Nf[sm.R+1:end]/sum(Nf[sm.R+1:end])
            NbR = Nb[sm.R+1:end]/sum(Nb[sm.R+1:end])
        else
            NfR = [1]
            NbR = [1]
        end

        S[i,1] = NfR'*(-hf.*tanh.(hf).+log.(2 .*cosh.(hf)))

        hf2 = hf*ones(1,Ncap(sm)-R)
        hb2 = ones(Ncap(sm)-R,1)*hb'
        Sr2 = (-hb2.*tanh.(hf2).+log.(2 .*cosh.(hb2)))
        
        S[i,2] = NfR'*Sr2*NbR
    end
    return S
end

# function entropy_old!(sm::IsingModel, meas_stp::Int, parallel :: Bool = true)
#     #get trajectory
#     fdist = fdist_traj!(sm, meas_stp+2, parallel=parallel)
#     # calculate entropy
#     S = zeros(meas_stp, 2)
#     for i in 1:meas_stp
#         Nf = fdist[i, 1, :]
#         Nb = fdist[i+2, 2, :] 

#         hf = [sm.β*(local_current(sm.J,τ,N2a(Nf,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]
#         hb = [sm.β*(local_current(sm.J,τ,N2a(Nb,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]

#         S[i,1] = Nf[1:sm.Q]'*(-hf.*tanh.(hf).+log.(2 .*cosh.(hf)))
#         S[i,2] = Nb[1:sm.Q]'*(-hb.*tanh.(hf).+log.(2 .*cosh.(hb)))
#     end
#     return S
# end

# Useful Graphs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #