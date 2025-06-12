using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));   
    using ProgressBars
    using CairoMakie
end







# plot with refracion

# MF: entropy along β
begin
    J = 0.2
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
    display(f)
end

# Eigenvalues along β
begin
    J = 0.2
    θ = 1
    R = 3
    I = 0.1
    Q = 50
    α = 0.1
    βs = range(0, 60, 51)
    ic = spike_ic_mf(R, Q)
    C = exponential_weights(Q, α)
    
    # run the simulations
    meas = zeros(length(βs), R+Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = CombinedIMF(ic, J, θ, βs[i], I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\beta", ylabel=L"\lambda")
    for i in 1:R+Q+1
        lines!(ax, βs, meas[:,i], color=:black, linewidth=1.5)
    end
    ylims!(ax, 0.9, 1.1)
    ax.title = L"Combined model, $\theta = %$θ; R = %$R; Q = %$Q; \alpha=%$α$"
    display(f)
end
