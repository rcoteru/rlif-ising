using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));     
end

using Distributions: Binomial
using Statistics: cor, std
using ProgressBars
using CairoMakie
using FFTW

using DataStructures

# 0D (time series)
begin
    N = 400
    J = 6
    R = 4
    θ = -1.3
    β = 1.35
    #s,n = spike_ic(N)
    s,n = random_ic(N)
    sm = UniformSM(N, J, R, s, n, θ, β);

    equi_stp = 3000
    meas_stp = 1000
    parallel = false

    forward!(sm, equi_stp, parallel=parallel)

    # traj = zeros(Float16, meas_stp*N)
    # for i in ProgressBar(1:meas_stp)
    #     perm = randperm(sm.N)
    #     for j in perm
    #         glauber_step!(sm, j)
    #         traj[(i-1)*N+j] = (mean(sm.s)+1)/2
    #     end
    # end

    traj = trajectory!(sm, meas_stp, parallel=parallel)
    traj = (traj.+1)./2


end;
# begin
#     fig = Figure()
#     nshow =100
#     ax = Axis(fig[1,1], xlabel="Time", ylabel="Activity")
#     x = 1/N:1/N:nshow
#     scatter!(x, traj[1:nshow*N], color=:blue, markersize=4)
#     fig
# end

begin
    fig = Figure()
    nshow =1000
    ax = Axis(fig[1,1], xlabel="Time", ylabel="Activity")
    x = 1:nshow
    lines!(x, traj[1:nshow], color=:blue)
    #scatter!(x, traj[1:nshow], color=:blue, markersize=4)
    fig
end


avals = []

function find_avalanches(traj::Vector, thold::Real, lfilt::Int=1)
    # avalanches are contiguous region of activity
    # find all avalanche start/end points
    
    av_pts = findall(traj .> thold)



end


function spike_stats(traj::Vector, 
        thold::Real, lfilt::Int=1)

    # avalanches are contiguous region of activity
    # find all avalanche start/end points
    
    av_pts = findall(traj .> thold)
    avalanches = Vector{Vector{Int}}([])
    push!(avalanches, [av_pts[1]])
    for (idx, prev_idx) in zip(av_pts[2:end], av_pts[1:end-1])
        if idx == prev_idx + 1
            push!(avalanches[end], idx)
        else
            push!(avalanches, [idx])
        end
    end
    # filter out avalanches of size 1
    avalanches = filter(x -> length(x) > lfilt, avalanches)

    # get avalanche durations
    av_len = [length(av) for av in avalanches]

    # get avalanche sizes
    av_sum = [sum(traj[av]) for av in avalanches]

    # get isi
    isi = zeros(Int, length(avalanches)-1)
    for i in 1:length(avalanches)-1
        isi[i] = avalanches[i+1][1] - avalanches[i][end]
    end

    return avalanches, av_sum, av_len, isi
end

avalanches, av_sum, av_len, isi = spike_stats(traj, 0.2, 2)

begin
    fig = Figure()
    ax = Axis(fig[1,1], yscale=log, xlabel="ISI", ylabel="Count")
    hist!(ax, isi, bins=25)
    fig
end


# avalanche area
# avalanche is contiguous region of activity
begin

    thold = 50/N
    spikes = findall(traj .> thold)
    
    spike_sizes = Vector{Float32}()
    push!(spike_sizes, traj[1])

    spikes = findall(traj .> thold)
    for (idx, prev_idx) in zip(spikes[2:end], spikes[1:end-1])
        if idx == prev_idx + 1
            spike_sizes[end] += traj[idx]
        else
            push!(spike_sizes, traj[idx])
        end
    end

   
    # for i in logrange(1, maximum(spike_sizes), 10)
    #     if haskey(spike_counts, i)
    #         spike_counts[i] += 1
    #     else
    #         spike_counts[i] = 1
    #     end
    # end

    # get counts of each isi
    spike_counts = Dict()
    
    for i in spike_sizes
        if haskey(spike_counts, i)
            spike_counts[i] += 1
        else
            spike_counts[i] = 1
        end
    end

    # convert to array
    x = [k for (k,v) in spike_counts]
    y = [v for (k,v) in spike_counts]

    f = Figure()
    ax = Axis(f[1,1], xlabel="Spike Size", ylabel="Count",
        yscale=log, xscale=log)
    scatter!(ax, x, y, color=:blue)
    f
end



traj[findall(traj .> thold)]
spike_sizes


logrange(1, maximum(spike_sizes), 20)


begin
    fig = Figure()
    ax = Axis(fig[1,1],yscale=log, xscale=log)
    hist!(ax, spike_sizes, bins=10)
    fig
end


begin
    tmax = 50
    corrs = [cor(traj[1:end-i], traj[1+i:end]) for i in 0:tmax]
    lines(0:tmax, corrs, color=:blue)
end
begin # fourier spectrum
    f = rfft(traj)
    f = abs.(f).^2
    f = f ./ sum(f)
    freqs = rfftfreq(length(traj))
    lines(freqs, f, color=:blue)
end
begin
    thold = 0.3
    traj

    traj .> thold
    spike_intervals = findall(traj .> thold)
    isi = diff(spike_intervals)
    isi = isi[isi .> 1]

    # get counts of each isi
    isi_counts = Dict()
    for i in isi
        if haskey(isi_counts, i)
            isi_counts[i] += 1
        else
            isi_counts[i] = 1
        end
    end
    # convert to array
    x = [k for (k,v) in isi_counts]
    y = [v for (k,v) in isi_counts]


    f = Figure()
    ax = Axis(f[1,1], xlabel="ISI", ylabel="Count",
        yscale=log, xscale=log)
    scatter!(ax, x, y, color=:blue)
    f
end
