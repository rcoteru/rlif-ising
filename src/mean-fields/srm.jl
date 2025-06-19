include("../discrete-maps.jl")

import Distributions: Dirichlet
using FFTW



function mf_pots(x, p, pc)
    return [pc[:κ][1:τ]'*(p[:J]*x[1:τ] .+ p[:I]) for τ in 1:p[:Q]] + pc[:η]
end

function mf_map(x, p, pc)
    activ = tanh.(p[:β]*(mf_pots(x,p,pc).-p[:θ]))./2
    return [
        x[1:p[:Q]]'*(0.5.+activ) + x[p[:Q]+1]'*(0.5.+activ[p[:Q]]),
        x[1:p[:Q]-1].*(0.5.-activ[1:p[:Q]-1])...,
        (x[p[:Q]] + x[p[:Q]+1])*(0.5.-activ[p[:Q]])
    ]
end

function SRM_mf(x0::Vector,
    J::Real, θ::Real, β::Real, I::Real,
    τm::Real, K::Real
    ) :: DiscreteMap
    # deduce Q from x0
    Q = length(x0) - 1
    # create parameter dictionary
    p = Dict(:J => J,:θ => θ, :β => β, :I => I, 
        :Q => Q, :τm => τm, :K => K)
    # create featurizer
    ft = (x, p, pc) -> [
        x[1],               # firing neurons
        sum(x[1:p[:Q]]),    # unsaturated neurons
        sum(x[p[:Q]+1]),    # saturated neurons
        norm(x)             # norm of the state (should be 1)
    ]
    nft = 4
    # precompute stuff
    pc = Dict(
        :κ => exp.(-(0:p[:Q]-1)/p[:τm]),
        :η => -p[:K]*exp.(-(0:p[:Q]-1)/p[:τm]),
    )
    return DiscreteMap{Q+1}("SRM Mean Field",
        mf_map, x0, x0, p, ft, nft, pc)
end

function run_ts(ic, J, θ, I, β, τm, K, nequi, nmeas)
    dm = SRM_mf(ic, J, θ, β, I, τm, K)
    forward!(dm, nequi)
    return trajectory!(dm, nmeas, ft=true)[:,1]
end

# get main frequency from the trajectory
function get_main_freq(ts)
    # compute the FFT of the first time series
    fft_result = rfft(ts)[2:end]  # skip the DC component
    # find the index of the maximum amplitude
    max_index = argmax(abs.(fft_result))
    # if the maximum index is 0, return 0 frequency
    if max_index == 0
        return 0
    end
    # calculate frequency vector
    return rfftfreq(size(ts, 1), 1000)[2:end][max_index]
end

function fdist_traj!(dm::DiscreteMap, meas_stp::Int)
    traj_size = meas_stp + 2*dm.p[:Q]
    tf = trajectory!(dm, traj_size, ft=false)
    Q, scan = dm.p[:Q], dm.p[:Q]
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

function SRM_entropy(dm::DiscreteMap, meas_stp::Int)
    fdist = fdist_traj!(dm, meas_stp+2)
    S, Q = zeros(meas_stp,2), dm.p[:Q]
    for i in 1:meas_stp
        # p(n) / p(\hat{n})
        Nf = fdist[i,1,:]
        Nb = fdist[i+2,2,:]
        # local fields
        hf = dm.p[:β].*(mf_pots(Nf,dm.p,dm.pc).-dm.p[:θ])
        hb = dm.p[:β].*(mf_pots(Nb,dm.p,dm.pc).-dm.p[:θ])
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Initial conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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