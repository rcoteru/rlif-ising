using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
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
    J = 1
    R = 3
    θ = 0
    I = 0
    βs = range(0, 10, 101)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 1000
    
    # run the simulations
    meas = zeros(length(βs), 12)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([βs]))
        dm = RefractiveIMF(ic, J, θ, βs[i], I)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange, :purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4]
        lines!(ax, βs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, βs, meas[:,i]+meas[:,i+6], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas[:,i]-meas[:,i+6], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std along θ
begin
    J = 1
    R = 3
    β = 50
    I = 0
    θs = range(-0.5,0.5, 101)
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 40000, 20000
    
    # run the simulations
    meas = zeros(length(θs), 12)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = RefractiveIMF(ic, J, θs[i], β, I)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange, :purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4]
        lines!(ax, θs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, θs, meas[:,i]+meas[:,i+6], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas[:,i]-meas[:,i+6], color = colors[i], linestyle = :dash)
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
    I = 0
    βs = range(0, 50, 101)
    θs = range(-1,1, 101)
    ic = quiet_ic_mf(R, 0)

    nequi, nmeas = 2000, 1000
    
    # run the simulations
    meas = zeros(length(θs), length(βs), 12)
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j], I)
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
    Colorbar(f[4,2], hm, label = "Fxd")

    # kuramoto
    ax = Axis(f[5,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, meas[:,:,12]', colormap = :viridis)
    Colorbar(f[5,2], hm, label = "phiK std")

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
    J = 1
    R = 3
    θ = 0
    βs = range(0, 10, 101)
    I = 0
    ic = random_ic_mf(R, 0)

    nequi, nmeas = 2000, 3000
    
    # run the simulations
    ent = zeros(length(βs), 2)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = RefractiveIMF(ic, J, θ, βs[i], I)
        forward!(dm, nequi)
        S = refractive_entropy!(dm, nmeas)
        ent[i,:] = mean(S, dims = 1) 
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
    lines!(ax, βs, ent[:,2]-ent[:,1], color = :black)
    ax.title = L"Total entropy$$"

    save(plotsdir("refractive-mf-entropy-beta.pdf"), f)
    display(f)
end

# MF: entropy along θ
begin
    J = 1
    R = 3
    β = 10
    θs = range(-2, 2, 101)
    I = 0
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 5000
    
    # run the simulations
    ent = zeros(length(θs), 2)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = RefractiveIMF(ic, J, θs[i], β, I)
        forward!(dm, nequi)
        S = refractive_entropy!(dm, nmeas)
        ent[i,:] = mean(S, dims = 1)
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
    lines!(ax, θs, ent[:,2]-ent[:,1], color = :black)
    ax.title = L"Total entropy $$"

    save(plotsdir("refractive-mf-entropy-theta.pdf"), f)
    display(f)
end

# MF: entropy in β-θ plane
begin
    J = 1
    R = 5
    βs = range(0, 6, 51)
    θs = range(-1, 2, 51)
    I = 0
    ic = spike_ic_mf(R, 0)

    nequi, nmeas = 20000, 1000
    
    # run the simulations
    ent = zeros(length(θs), length(βs),2)
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j], I)
        forward!(dm, nequi)
        S = refractive_entropy!(dm, nmeas)
        ent[i,j,:] = mean(S, dims = 1)
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
    hm = heatmap!(ax, βs, θs, ent[:,:,2]'-ent[:,:,1]', 
        colormap = :viridis)
    Colorbar(f[3,2], hm, label = "Total Entropy")

    save(plotsdir("refractive-mf-entropy-plane.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# SM: mean, std along β
begin
    N = 200
    J = 1
    R = 3
    θ = 0
    I = 0
    βs = range(0, 10, 101)
    
    ic = spike_ic_sm(N)

    nequi, nmeas = 100, 1000
    parallel = true
    
    # run the simulations
    meas = zeros(length(βs), 10)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([βs]))    
        sm = RefractiveSM(J, θ, βs[i], I, ic, R)
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
        lines!(ax, βs, meas[:,i]+meas[:,i+5], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas[:,i]-meas[:,i+5], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ$"
    axislegend(ax, position = :lt)
    save(plotsdir("stability-line.pdf"), f)
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

    nequi, nmeas = 100, 1000
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

    nequi, nmeas = 5000, 2000
    
    # run the simulations
    meas = zeros(length(βs), 12)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
            dm = IntegratorIMF(ic, J, θ, βs[i], I, C)
            forward!(dm, nequi)
            meas[i,:] = stats!(dm, nmeas, ft=true)  
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange, :purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4,5,6]
        lines!(ax, βs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, βs, meas[:,i]+meas[:,i+6], color = colors[i], linestyle = :dash)
        lines!(ax, βs, meas[:,i]-meas[:,i+6], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\theta = %$θ$"
    #ylims!(ax, 0, 1)
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
    meas = zeros(length(θs), 12)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = IntegratorIMF(ic, J, θs[i], β, I, C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"θ", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange,:purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4,5,6]
        lines!(ax, θs, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, θs, meas[:,i]+meas[:,i+6], color = colors[i], linestyle = :dash)
        lines!(ax, θs, meas[:,i]-meas[:,i+6], color = colors[i], linestyle = :dash)
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
    meas = zeros(length(αs), 12)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([αs]))    
        C = exponential_weights(Q, αs[i])
        dm = IntegratorIMF(ic, J, θ, β, I, C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"α", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange,:purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4,5,6]
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
    meas = zeros(length(Js), 12)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([Js]))    
        dm = IntegratorIMF(ic, Js[i], θ, β, I, C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"J", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange,:purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4,5,6]
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

    nequi, nmeas = 10000, 5000
    
    # run the simulations
    meas = zeros(length(Is), 12)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([Is]))    
        dm = IntegratorIMF(ic, J, θ, β, Is[i], C)
        forward!(dm, nequi)
        meas[i,:] = stats!(dm, nmeas, ft=true)
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"I", ylabel=L"n")
    colors = [:black, :red, :blue, :green, :orange,:purple]
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1,4,5,6]
        lines!(ax, Is, meas[:,i], color = colors[i], label = labels[i])
        lines!(ax, Is, meas[:,i]+meas[:,i+6], color = colors[i], linestyle = :dash)
        lines!(ax, Is, meas[:,i]-meas[:,i+6], color = colors[i], linestyle = :dash)
    end
    # set title with θ dynamically
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β$"
    axislegend(ax, position = :lt)
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# MF: mean, std in β-θ plane
begin
    J = 0.1
    βs = range(0, 50, 51)
    θs = range(0.5, 1.5, 51)
    I = 0.1
    α = 0.1
    Q = 50
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)

    nequi, nmeas = 10000, 10000
end
begin
    # run the simulations
    meas = zeros(length(θs), length(βs), 12)
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = IntegratorIMF(ic, J, θs[i], βs[j], I, C)
        forward!(dm, nequi)
        meas[i,j,:] = stats!(dm, nmeas, ft=true)
    end
end
begin  # plot 
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
    Colorbar(f[4,2], hm, label = "Fxd")

    # kuramoto r
    ax = Axis(f[5,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, meas[:,:,5]', colormap = :viridis)
    Colorbar(f[5,2], hm, label = "rK")

    # kuramoto phi
    ax = Axis(f[6,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, meas[:,:,12]', colormap = :viridis)
    Colorbar(f[6,2], hm, label = "phiK std")

    save(plotsdir("integrator-mf-stats-plane-mean.pdf"), f)
    display(f) 
end

# MF: fdist
begin
    J = 0.2
    θ = 1
    β = 30
    I = 0.1
    α = 0.1
    Q = 50
    ic = random_ic_mf(0, Q)
    C = exponential_weights(Q, α)

    nequi, nmeas = 2000, 3000
    dm = IntegratorIMF(ic, J, θ, β, I, C)
    forward!(dm, nequi)
    fdist = integrator_fdist_traj!(dm, nmeas)

    idx = 50
    f = Figure(size = (800, 400))
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"p")
    lines!(ax, 1:Q+1, fdist[idx,1,:], color = :blue, 
        label=L"p(n)")
    lines!(ax, 1:Q+1, fdist[idx,2,:], color = :red,
        label=L"p(\hat{n})")
    axislegend(ax, position = :lt)
    display(f)

end


# MF: entropy along β
begin
    J = 0.3
    θ = 1
    βs = range(0, 30, 31)
    I = 0.1
    α = 0.1
    Q = 50
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)

    nequi, nmeas = 10000, 4000
    
    # run the simulations
    ent = zeros(length(βs), 2)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = IntegratorIMF(ic, J, θ, βs[i], I, C)
        forward!(dm, nequi)
        S = integrator_entropy!(dm, nmeas)
        ent[i,:] = mean(S, dims = 1) 
    end
end
begin
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
    lines!(ax, βs, ent[:,2]-ent[:,1], color = :black)
    ax.title = L"Total entropy$$"
    save(plotsdir("integrator-mf-entropy-beta.pdf"), f)
    display(f)
end

# MF: entropy along θ
begin
    J = 0.1
    β = 20
    θs = range(0, 1.5, 51)
    I = 0.1
    α = 0.1
    Q = 50
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)

    nequi, nmeas = 10000, 2000
    
    # run the simulations
    ent = zeros(length(θs), 2)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = IntegratorIMF(ic, J, θs[i], β, I, C)
        forward!(dm, nequi)
        S = integrator_entropy!(dm, nmeas)
        ent[i,:] = mean(S, dims = 1)
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
    lines!(ax, θs, ent[:,2]-ent[:,1], color = :black)
    ax.title = L"Total entropy $$"
    save(plotsdir("integrator-mf-entropy-theta.pdf"), f)
    display(f)
end