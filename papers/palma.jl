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
    I = 0.025
    β = 20
    Q = 50
    τm = 10
    K = 0.3

    nequi, nmeas = 10000, 200

    ic = random_ic_mf(0, Q)
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
    axislegend(ax, position = :rt, title = "Legend")
    display(fig)
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
    J = 0.1
    θ = 1
    I_values = range(0, stop=0.15, length=21)
    β = 40
    Q = 50
    τm = 10
    K = 0.1
    nequi, nmeas = 10000, 2000
    freqs = zeros(Float32,(length(I_values)))
    @showprogress Threads.@threads for i in eachindex(I_values)
        ts = run_ts(J, θ, I_values[i], β, Q, τm, K, nequi, nmeas)
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


# run 