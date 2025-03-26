using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field2.jl"));
    include(srcdir("auxiliary.jl"));   
    using ProgressBars
    using CairoMakie
    using FFTW
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Refractive model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# MF: mean, std along β
begin
    J = -1
    R = 3
    θ = 0
    βs = range(0, 50, 101)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 1000
    
    # run the simulations
    meas = zeros(length(βs), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = RefractiveIMF(ic, J, θ, βs[i])
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1]
        lines!(ax, βs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, βs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std along θ
begin
    J = 0.1
    R = 3
    β = 50
    θs = range(-0.5,0.5, 101)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 30000
    
    # run the simulations
    meas = zeros(length(θs), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = RefractiveIMF(ic, J, θs[i], β)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, θs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, θs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std in β-θ plane
begin
    J = -1
    R = 3
    βs = range(0, 50, 101)
    θs = range(-1,1, 101)
    ic = quiet_ic_mf(R, 0)

    nequi, nmeas = 2000, 1000
    
    # run the simulations
    meas = zeros(length(θs), length(βs), 8)
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j])
        forward!(dm, nequi)
        meas[i,j,:] = stats!(dm, nmeas, ft=true)
    end

end
begin
    # plot
    f = Figure(size = (800, 1600))

    # activity
    ax = Axis(f[1,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, meas[:,:,1]', colormap = :viridis)
    Colorbar(f[1,2], hm, label = "Activity")

    # refractive
    ax = Axis(f[2,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, meas[:,:,2]', colormap = :viridis)
    Colorbar(f[2,2], hm, label = "Refractive")

    # ready
    ax = Axis(f[3,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, meas[:,:,3]', colormap = :viridis)
    Colorbar(f[3,2], hm, label = "Ready")

    # fxd distance
    ax = Axis(f[4,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, meas[:,:,4]', colormap = :viridis)
    Colorbar(f[4,2], hm, label = "Fxd distance")

    save(plotsdir("refractive-mf-stats-plane-mean.pdf"), f)
    display(f)
end
begin
    # plot
    f = Figure(size = (800, 1600))

    # activity
    ax = Axis(f[1,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, meas[:,:,5]', colormap = :viridis)
    Colorbar(f[1,2], hm, label = "Activity")

    # refractive
    ax = Axis(f[2,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, meas[:,:,6]', colormap = :viridis)
    Colorbar(f[2,2], hm, label = "Refractive")

    # ready
    ax = Axis(f[3,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, meas[:,:,7]', colormap = :viridis)
    Colorbar(f[3,2], hm, label = "Ready")

    # fxd distance
    ax = Axis(f[4,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, meas[:,:,8]', colormap = :viridis)
    Colorbar(f[4,2], hm, label = "Fxd distance")

    save(plotsdir("refractive-mf-stats-plane-std.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# MF: entropy along β
begin
    J = -1
    R = 3
    θ = 0
    βs = range(0, 20, 101)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 1000
    
    # run the simulations
    ent = zeros(length(βs), 3)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = RefractiveIMF(ic, J, θ, βs[i])
        forward!(dm, nequi)
        ent[i,:] = refractive_ising_entropy!(dm, nmeas)
    end

    # plot
    f = Figure(size = (800, 400))

    # forwards entropy
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"S")
    lines!(ax, βs, ent[:,1], color = :black)
    ax.title = L"Forwards entropy$$"

    # backwards entropy
    ax = Axis(f[1,2], xlabel=L"β")
    lines!(ax, βs, ent[:,2], color = :black)
    ax.title = L"Backwards entropy, $\theta = %$θ$"

    # total entropy
    ax = Axis(f[1,3], xlabel=L"β")
    lines!(ax, βs, ent[:,3], color = :black)
    ax.title = L"Total entropy$$"

    save(plotsdir("refractive-mf-entropy-beta.pdf"), f)
    display(f)
end

# MF: entropy along θ
begin
    J = -1
    R = 3
    β = 10
    θs = range(-2, 2, 101)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 1000
    
    # run the simulations
    ent = zeros(length(θs), 3)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = RefractiveIMF(ic, J, θs[i], β)
        forward!(dm, nequi)
        ent[i,:] = refractive_ising_entropy!(dm, nmeas)
    end

    # plot
    f = Figure(size = (800, 400))
    
    # forwards entropy
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"S")
    lines!(ax, θs, ent[:,1], color = :black)
    ax.title = L"Forwards entropy$$"

    # backwards entropy
    ax = Axis(f[1,2], xlabel=L"θ")
    lines!(ax, θs, ent[:,2], color = :black)
    ax.title = L"Backwards entropy, $\beta = %$β$"

    # total entropy
    ax = Axis(f[1,3], xlabel=L"θ")
    lines!(ax, θs, ent[:,3], color = :black)
    ax.title = L"Total entropy $$"

    save(plotsdir("refractive-mf-entropy-theta.pdf"), f)
    display(f)
end

# MF: entropy in β-θ plane
begin
    J = 1
    R = 3
    βs = range(0, 10, 51)
    θs = range(-2, 2, 51)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 1000
    
    # run the simulations
    ent = zeros(length(θs), length(βs), 3)
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j])
        forward!(dm, nequi)
        ent[i,j,:] = refractive_ising_entropy!(dm, nmeas)
    end
end
begin
    # plot
    f = Figure(size = (400, 600))

    # forwards entropy
    ax = Axis(f[1,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, ent[:,:,1]', colormap = :viridis)
    Colorbar(f[1,2], hm, label = "Forwards Entropy")

    # backwards entropy
    ax = Axis(f[2,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, ent[:,:,2]', colormap = :viridis)
    Colorbar(f[2,2], hm, label = "Backwards Entropy")

    # total entropy
    ax = Axis(f[3,1], ylabel=L"\theta")
    hm = heatmap!(ax, βs, θs, ent[:,:,3]', colormap = :viridis)
    Colorbar(f[3,2], hm, label = "Total Entropy")

    save(plotsdir("refractive-mf-entropy-plane.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# SM: mean, std along β
begin
    N = 2000
    J = 1
    R = 3
    θ = 0
    βs = range(0, 10, 101)
    
    ic = spike_ic_sm(N)

    nequi, nmeas = 1000, 10000
    parallel = true
    
    # run the simulations
    meas = zeros(length(βs), 8)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([βs]))    
        sm = RefractiveSM(J, θ, βs[i], ic, R)
        forward!(sm, nequi, parallel=parallel)
        meas[i,:] = stats!(sm, nmeas, parallel=parallel)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, βs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, βs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# SM: mean, std along θ
begin
    N = 2000
    J = 1
    R = 3
    β = 5
    θs = range(-1,1, 101)
    
    ic = spike_ic_sm(N)

    nequi, nmeas = 1000, 2000
    parallel = true
    
    # run the simulations
    meas = zeros(length(θs), 8)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([θs]))    
        sm = RefractiveSM(J, θs[i], β, ic, R)
        forward!(sm, nequi, parallel=parallel)
        meas[i,:] = stats!(sm, nmeas, parallel=parallel)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, θs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, θs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# MF vs SM: mean, std along β
begin
    N = 2000
    J = 1
    R = 3
    θ = 0
    βs = range(0, 10, 101)
    
    ic = spike_ic_sm(N)

    nequi, nmeas = 1000, 5000
    parallel = true
    
    # run the simulations
    meas_sm = zeros(length(βs), 8)
    meas_mf = zeros(length(βs), 8)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([βs]))    
        sm = RefractiveSM(J, θ, βs[i], ic, R)
        forward!(sm, nequi, parallel=parallel)
        meas_sm[i,:] = stats!(sm, nmeas, parallel=parallel)
        
        mf = RefractiveIMF(spike_ic_mf(R, 0), J, θ, βs[i])
        forward!(mf, nequi)
        meas_mf[i,:] = stats!(mf, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, βs, meas_sm[:,i], color = colors[i], label = labels[i]*" SM")
        lines!(ax, βs, meas_sm[:,i]+meas_sm[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas_sm[:,i]-meas_sm[:,i+4], color = colors[i], linestyle = :dash)
        
        lines!(ax, βs, meas_mf[:,i], color = colors[i], label = labels[i]*" MF")
        lines!(ax, βs, meas_mf[:,i]+meas_mf[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas_mf[:,i]-meas_mf[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ$"
    axislegend(ax, position = :lt)
    save(plotsdir("refractive-mf-sm-stats-beta.pdf"), f)
    display(f)
end

# MF vs SM: mean, std along θ
begin
    N = 1000
    J = 1
    R = 3
    β = 5
    θs = range(-1,1, 101)
    
    ic = spike_ic_sm(N)

    nequi, nmeas = 1000, 5000
    parallel = true
    
    # run the simulations
    meas_sm = zeros(length(θs), 8)
    meas_mf = zeros(length(θs), 8)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([θs]))    
        sm = RefractiveSM(J, θs[i], β, ic, R)
        forward!(sm, nequi, parallel=parallel)
        meas_sm[i,:] = stats!(sm, nmeas, parallel=parallel)
        
        mf = RefractiveIMF(spike_ic_mf(R, 0), J, θs[i], β)
        forward!(mf, nequi)
        meas_mf[i,:] = stats!(mf, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, θs, meas_sm[:,i], color = colors[i], label = labels[i]*" SM")
        lines!(ax, θs, meas_sm[:,i]+meas_sm[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas_sm[:,i]-meas_sm[:,i+4], color = colors[i], linestyle = :dash)
        
        lines!(ax, θs, meas_mf[:,i], color = colors[i], label = labels[i]*" MF")
        lines!(ax, θs, meas_mf[:,i]+meas_mf[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas_mf[:,i]-meas_mf[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with beta dynamically
    ax.title = L"Refractive model, $\beta = %$β$"
    axislegend(ax, position = :lt)
    save(plotsdir("refractive-mf-sm-stats-theta.pdf"), f)
    display(f)

end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Integrator model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Visualize fixed point
begin
    J = 0.1
    θ = 1
    β = 10
    Q = 50
    I = 0.1
    α = 0.1
    C = exponential_weights(Q, α)

    ic = spike_ic_mf(0, Q)
    #ic = quiet_ic_mf(0, Q)
    fxp = integrator_fxp(J, θ, β, I, C)


    dm = IntegratorIMF(ic, J, θ, β, I, C)
    forward!(dm, 1000)


    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 1:Q+1, fxp, color = :black, label="Theoretical")
    lines!(ax, 1:Q+1, dm.pc[:fxp], color = :red, label="Simulated")
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; Q = %$Q$"
    axislegend(ax, position = :lt)
    display(f)
end

# Visualize weights
begin
    Q = 50
    α = 0.1
    C = exponential_weights(Q, α)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"w_n")
    lines!(ax, 1:Q, C)
    ax.title = L"Exponential weights, $\alpha=%$α$"
    display(f)
end


# MF: mean, std along β
begin
    J = 0.1
    θ = 1
    βs = range(0, 50, 51)
    I = 0.1
    Q = 50
    C = exponential_weights(Q, 0.1)
    ic = spike_ic_mf(0, Q)
    #ic = quiet_ic_mf(0, Q)

    nequi, nmeas = 1000, 1000
    
    # run the simulations
    meas = zeros(length(βs), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
            dm = IntegratorIMF(ic, J, θ, βs[i], I, C)
            forward!(dm, nequi)
            meas[i,:] = stats!(dm, nmeas, ft=true)  
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, βs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, βs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\theta = %$θ$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std along θ
begin
    J = 0.1
    β = 20
    θs = range(0.5, 1.5, 51)
    I = 0.1
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(0, Q)

    nequi, nmeas = 10000, 2000
    
    # run the simulations
    meas = zeros(length(θs), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = IntegratorIMF(ic, J, θs[i], β, I, C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, θs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, θs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std along α
begin
    J = 0.1
    β = 30
    θ = 1
    αs = range(0, 0.2, 51)
    I = 0.1
    Q = 50
    ic = spike_ic_mf(0, Q)

    nequi, nmeas = 10000, 4000
    
    # run the simulations
    meas = zeros(length(αs), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([αs]))    
        C = exponential_weights(Q, αs[i])
        dm = IntegratorIMF(ic, J, θ, β, I, C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"α", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, αs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, αs, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, αs, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std along J 
begin
    Js = range(0, 1, 51)
    θ = 1
    β = 20
    α = 0.1
    I = 0.1
    Q = 50
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)

    nequi, nmeas = 10000, 10000
    
    # run the simulations
    meas = zeros(length(Js), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([Js]))    
        dm = IntegratorIMF(ic, Js[i], θ, β, I, C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"J", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, Js, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, Js, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, Js, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std along I
begin
    J = 0.1
    θ = 1
    β = 20
    α = 0.1
    Q = 50
    Is = range(0.05, 0.15, 51)
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)

    nequi, nmeas = 10000, 10000
    
    # run the simulations
    meas = zeros(length(I), 8)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([Is]))    
        dm = IntegratorIMF(ic, J, θ, β, Is[i], C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"I", ylabel=L"n")
    colors = [:black, :red, :blue, :green]
    labels = ["Firing", "Refractive", "Ready", "Fxp distance"]
    for i in [1,4]
        lines!(ax, I, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, I, meas[:,i]+meas[:,i+4], color = colors[i], linestyle = :dash)
        lines!(ax, I, meas[:,i]-meas[:,i+4], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end