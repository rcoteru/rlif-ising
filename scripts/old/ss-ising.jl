using DrWatson
@quickactivate "BioIsing"

include(srcdir("spin-models.jl"));
include(srcdir("convenience.jl"));

using Distributions: Binomial
using Statistics: cor, std
using ProgressBars
using CairoMakie

using FFTW


# Uniform spin model (no integration)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 0D (time series)
begin
    N = 500
    J = 6
    R = 4
    θ = -0.51
    β = 5
    #s,n = spike_ic(N)
    s,n = random_ic(N)
    sm = UniformSM(N, J, R, s, n, θ, β);

    equi_stp = 2000
    meas_stp = 100000
    parallel = true

    forward!(sm, equi_stp, parallel=parallel)
    traj = trajectory!(sm, meas_stp, parallel=parallel)
    traj = (traj.+1)./2
end;
begin
    fig = Figure()
    nshow = 5000
    ax = Axis(fig[1,1])
    lines!(1:nshow, traj[1:nshow], color=:blue)
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


# 1D (β slice)
begin
    N = 200
    J = 6
    R = 4
    θ = -0.5
    βs = range(0, 5, length=51)

    #s,n = spike_ic(N)
    s,n = random_ic(N)

    equi_stp = 1000
    meas_stp = 10000

    parallel = true
    active = true

    meas = zeros(length(βs));
    for (i,) in ProgressBar(idx_combinations([βs]))
        sm = UniformSM(N, J, R, s, n, θ, βs[i])
        forward!(sm, equi_stp,
            parallel=parallel, active=active)
        meas[i] = std(trajectory!(sm, meas_stp, 
            parallel=active, active=active))
    end
end;

begin
    fig = Figure()
    ax = Axis(fig[1,1], xlabel=L"β", ylabel="m")
    lines!(βs, meas, color=:blue)
    fig
end


# 2D (phase map)
begin
    N = 200
    J = 1
    R = 4
    θs = range(-2,2, length=51)
    βs = range(0, 6, length=51)

    #s,n = spike_ic(N)
    s,n = random_ic(N)

    equi_stp = 50
    meas_stp = 2000

    parallel = false
    active = true

    meas = zeros(length(θs), length(βs), 2);
    for (i,j) in ProgressBar(idx_combinations([θs, βs]))
        sm = UniformSM(N, J, R, s, n, θs[i], βs[j])
        forward!(sm, equi_stp,
            parallel=parallel, active=active)
        meas[i,j,:] = stats!(sm, meas_stp, 
            parallel=active, active=active)
    end
end;
begin
    fig = Figure()
    ax = Axis(fig[1,1], xlabel=L"θ", ylabel=L"β")
    hm = heatmap!(ax, βs, θs, meas[:,:,2]', colormap=:viridis)
    Colorbar(fig[1,2], hm)
    fig
end


# Uniform spin model (with integration)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

N = 100
Q = 20

J = 6
α = 0
R =  4
θ = -0.51
β = 5

s, n = spike_ic(N)
C = exp.(-α*(1:Q))/exp.(-α)

m = [mean(s), zeros(Q-1)...]

C'*m

Q = 20
C = exp.(-α*(1:Q))/exp.(-α)

m = zeros(Int, Q)

sm = IntUniformSM(N, Q, J, R, C, s, n, m, θ, β)


function local_field(s:: IntUniformSM, active::Bool = true) :: Float64
    if active
        return s.J*C'*((m.+1)./2) + s.θ
    end
    return s.J*mean(s.s) + s.θ
end


# 0D (time series)
begin
    N = 500
    J = 6
    R = 4
    θ = -0.51
    β = 5
    #s,n = spike_ic(N)
    s,n = random_ic(N)
    sm = UniformSM(N, J, R, s, n, θ, β);

    equi_stp = 2000
    meas_stp = 100000

    parallel = true
    active = true

    forward!(sm, equi_stp, 
    parallel=parallel, active=active)
 
    traj = trajectory!(sm, meas_stp, 
        parallel=active, active=active)
    traj = (traj.+1)./2
end;