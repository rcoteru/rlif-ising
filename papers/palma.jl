using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("mean-fields/srm.jl"));
    include(srcdir("auxiliary.jl"));
    using ProgressMeter
    using CairoMakie
    CairoMakie.activate!()
end

# MF: Time series 
begin
    J = 0.2
    θ = 1
    I = 0.1
    β = 50
    Q = 100
    τm = 10
    K = 1

    nequi, nmeas = 20000, 10000

    ic = spike_ic_mf(0, Q)
    #ic = spike_ic_mf(0, Q)
    dm = SRM_mf(ic, J, θ, β, I, τm, K)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Time", ylabel = "Activity", title = "MF Time Series")
    lines!(ax, traj[:,1], color = :blue, label = "Firing Neurons")
    #lines!(ax, traj[:,2], color = :red, label = "Unsaturated Neurons")
    #lines!(ax, traj[:,3], color = :green, label = "Saturated Neurons")
    #lines!(ax, traj[:,4], color = :black, label = "Norm")
    xlims!(ax, 0, size(traj, 1))
    ylims!(ax, 0, 1)    
    axislegend(ax, position = :rt, title = "Legend")
    display(fig)
end


using StatsBase: autocor

lines(autocor(traj[:,1], 1:200, demean=true))


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

# run same simulation with different K
begin
    J = 0.2
    θ = 1
    I = 0.1
    β = 20
    Q = 50
    τm = 10
    ic = random_ic_mf(0, Q)
    nequi, nmeas = 10000, 2000
    K_values = range(0, stop=1, length=41)
    freqs = zeros(Float32,(length(K_values)))
    @showprogress Threads.@threads for (i,) in idx_combinations([K_values])
        ts = run_ts(ic, J, θ, I, β, τm, K_values[i], nequi, nmeas)
        freqs[i] = get_main_freq(ts)
    end
    # plot main frequency vs K
    f = Figure(resolution = (800, 600))
    ax = Axis(f[1, 1], xlabel = "K", ylabel = "Freq (hz)", title = "Main Freq vs K")
    lines!(ax, K_values, freqs)
    display(f)
end

# run same simulation with different β
begin
    J = 0.1
    θ = 1
    I = 0.1
    β_values = range(0, stop=41, length=61)
    Q = 50
    τm = 10
    K = 0.1
    ic = random_ic_mf(0, Q)
    nequi, nmeas = 10000, 2000
    freqs = zeros(Float32,(length(β_values)))
    @showprogress Threads.@threads for (i,) in idx_combinations([β_values])
        ts = run_ts(ic, J, θ, I, β_values[i], τm, K, nequi, nmeas)
        freqs[i] = get_main_freq(ts)
    end
    # plot main frequency vs β
    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], xlabel = "β", ylabel = "Freq (hz)", title = "Main Freq vs β")
    lines!(ax, β_values, freqs)
    display(f)
end


# entropy production along β
begin
    J = 0.2
    θ = 1
    I = 0.1
    β_values = range(0, stop=41, length=61)
    Q = 50
    τm = 10
    K = 0
    ic = random_ic_mf(0, Q)
    nequi, nmeas = 10000, 2000
    freqs = zeros(Float32,(length(β_values)),2)
    #@showprogress Threads.@threads 
    @showprogress for (i,) in idx_combinations([β_values])
    
        dm = SRM_mf(ic, J, θ, β_values[i], I, τm, K)
        forward!(dm, nequi)
        S = SRM_entropy(dm, nmeas)
        freqs[i,:] = mean(S, dims=1)
    end
end
begin
    # plot main frequency vs β
    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], xlabel = "β", ylabel = "Entropy production", title = "Main Freq vs β")
    lines!(ax, β_values, freqs[:,1], label = "Forwards")
    lines!(ax, β_values, freqs[:,2], label = "Backwards")
    lines!(ax, β_values, freqs[:,2] - freqs[:,1], label = "Total")
    axislegend(ax, title = "Legend")
    display(f)
end

# entropy production along I
begin
    J = 0.2
    θ = 1
    I_values = range(0, stop=1, length=51)
    β = 100
    Q = 50
    τm = 10
    K = 1
    ic = spike_ic_mf(0, Q)
    nequi, nmeas = 20000, 4000
    freqs = zeros(Float32,(length(I_values)),2)
    @showprogress Threads.@threads for (i,) in idx_combinations([I_values])
        dm = SRM_mf(ic, J, θ, β, I_values[i], τm, K)
        forward!(dm, nequi)
        S = SRM_entropy(dm, nmeas)
        freqs[i,:] = mean(S, dims=1)
    end
end
begin
    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], xlabel = "I", ylabel = "Entropy production", title = "Main Freq vs I")
    #lines!(ax, I_values, freqs[:,1], label = "Forwards")
    #lines!(ax, I_values, freqs[:,2], label = "Backwards")
    lines!(ax, I_values, freqs[:,2] - freqs[:,1], label = "Total")
    axislegend(ax, title = "Legend")
    display(f)
end



# run same simulation with different I
begin
    J = 0.2
    θ = 1
    I_values = range(0, stop=0.8, length=41)
    β = 40
    Q = 50
    τm = 10
    K = 0
    ic = random_ic_mf(0, Q)
    nequi, nmeas = 10000, 2000
    freqs = zeros(Float32,(length(I_values)))
    @showprogress Threads.@threads for (i,) in idx_combinations([I_values])
        ts = run_ts(ic,J, θ, β, I_values[i], τm, K, nequi, nmeas)
        freqs[i] = get_main_freq(ts)
    end
    # plot main frequency vs I
    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], xlabel = "I", ylabel = "Freq (hz)", title = "Main Freq vs I")
    lines!(ax, I_values, freqs)
    display(f)
end

# same simulation along I for different β
begin
    J = 0.1
    θ = 1
    I_values = range(0, stop=0.15, length=21)
    β_values = range(0, stop=41, length=61)
    Q = 50
    τm = 10
    K = 0.1
    nequi, nmeas = 10000, 2000
    freqs = zeros(Float32,(length(I_values), length(β_values)))
    @showprogress Threads.@threads for i in eachindex(I_values)
        for j in eachindex(β_values)
            ts = run_ts(J, θ, I_values[i], β_values[j], Q, τm, K, nequi, nmeas)
            freqs[i,j] = get_main_freq(ts)
        end
    end
    # plot main frequency vs I and β
    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], xlabel = "I", ylabel = "Freq (hz)", title = "Main Freq vs I and β")
    heatmap!(ax, I_values, β_values, freqs', colormap=:viridis)
    display(f)
end


begin
    J = 0.1
    θ = 1
    Q = 50
    τm = 10
    K = 0.5

    ic = random_ic_mf(0, Q)

    Is = range(0, stop=0.2, length=21)
    βs = range(0, 100, 51)
    
    nequi, nmeas = 10000, 2000

    # run the simulations
    meas = zeros(length(Is), length(βs), 2)
    @showprogress Threads.@threads for (i,j) in idx_combinations([Is, βs])    
        dm =  SRM_mf(ic, J, θ, βs[j], Is[i], τm, K)
        forward!(dm, nequi)
        S = SRM_entropy(dm, nmeas)
        meas[i,j,:] = mean(S, dims=1)
    end
end

# save the simulation results and parameters
using JLD2
begin
    save("palma_entropy.jld2", "meas", meas, "Is", Is, "βs", βs,
        "J", J, "θ", θ, "τm", τm, "K", K)
end



begin
    # plot the results
    f = Figure(size = (800, 1200))

    cmap = :viridis

    ax = Axis(f[1, 1])
    hm = contourf!(ax, βs, Is, meas[:,:,1]', 
        levels=30, colormap=cmap)
    tightlimits!(ax)
    Colorbar(f[1,2], hm, label = "Forwards entropy")
    ax.title = L"Entropy production: $J=%$J; \theta=%$θ; \tau_m=%$τm; K=%$K$"
    ax.ylabel = L"I^{ext}"


    ax = Axis(f[2, 1])
    hm = contourf!(ax, βs, Is, meas[:,:,2]',
        levels=30, colormap=cmap)
    tightlimits!(ax)
    Colorbar(f[2,2], hm, label = "Backwards entropy")
    ax.ylabel = L"I^{ext}"


    ax = Axis(f[3, 1])
    hm = contourf!(ax, βs, Is, meas[:,:,2]' - meas[:,:,1]', 
        levels=30, colormap=cmap)
    tightlimits!(ax)
    Colorbar(f[3,2], hm, label = "Total Entropy")

    ax.xlabel = L"\beta"
    ax.ylabel = L"I^{ext}"

    display(f)
end



begin
    J = 0.1
    θ = 1
    Q = 50
    τm = 10
    K = 0.5

    ic = random_ic_mf(0, Q)

    Is = range(0, stop=0.2, length=21)
    βs = range(0, 100, 51)
    
    nequi, nmeas = 10000, 2000

    # run the simulations
    vals = zeros(length(θs), length(βs), Q+1)
    nins = zeros(length(θs), length(βs))
    emax = zeros(length(θs), length(βs))
    @showprogress Threads.@threads for (i,j) in idx_combinations([Is, βs])    
        dm =  SRM_mf(ic, J, θ, βs[j], Is[i], τm, K)
        dm.x[:] = dm.pc[:fxp]
        meas[i,j,:] = abs.(eigvals(dm))
        nins[i,j] = sum(meas[i,j,:] .> 1.00001)
        emax[i,j] = maximum(meas[i,j,2:end])
    end
end
begin
    J = 0.1
    Q = 50
    θs = range(0, 2, 51)
    βs = range(0, 50, 51)
    I = 0.1
    α = 0.1
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)
    
    # run the simulations
    meas = zeros(length(θs), length(βs), Q+1)
    nins = zeros(length(θs), length(βs))
    emax = zeros(length(θs), length(βs))
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = IntegratorIMF(ic, J, θs[i], βs[j], I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,j,:] = abs.(eigvals(dm))
        nins[i,j] = sum(meas[i,j,:] .> 1.00001)
        emax[i,j] = maximum(meas[i,j,2:end])
    end
end
begin
    # plot
    f = Figure()
    ax = Axis(f[1,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, nins')
    cbar = Colorbar(f[1,2], hm, label=L"Unstable modes$$")
    ax.title = L"Integrator model$$"
    save(plotsdir("integrator-unstable-plane.pdf"), f)
    display(f)
end