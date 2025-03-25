using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field2.jl"));
    include(srcdir("auxiliary.jl"));   
    using ProgressBars
    using CairoMakie   
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Refractive model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Visualize fixed point
begin
    J = -1
    θ = 0
    β = 5
    R = 3
    fxp = refractive_ising_fxp(J, θ, β, R)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 1:R+1, fxp)
    #xlims!(ax, 1, 20)
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# Eigenvalues along β
begin 
    J = -1
    R = 3
    θ = 0
    βs = range(0, 50, 101)
    ic = spike_ic_mf(R, 0)
    
    # run the simulations
    meas = zeros(length(βs), R+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = RefractiveIMF(ic, J, θ, βs[i])
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = abs.(eigvals(dm))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"λ")
    for i in 1:R+1
        lines!(ax, βs, meas[:,i])
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ, R = %$R$"
    save(plotsdir("refractive-stability-beta.pdf"), f)
    display(f)
end

# Eigenvalues along θ
begin 
    J = -1
    R = 3
    θs = range(-1, 1, 101)
    β = 10
    ic = spike_ic_mf(R, 0)
    
    # run the simulations
    meas = zeros(length(θs), R+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = RefractiveIMF(ic, J, θs[i], β)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = abs.(eigvals(dm))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\theta", ylabel=L"\lambda")
    for i in 1:R+1
        lines!(ax, θs, meas[:,i])
    end
    ax.title = L"Refractive model, $\beta = %$β, R = %$R$"
    save(plotsdir("refractive-stability-theta.pdf"), f)
    display(f)
end

# Unstable eigenvalues in β-θ plane
begin
    J = -1
    R = 3
    θs = range(-1, 1, 101)
    βs = range(0, 20, 101)
    ic = spike_ic_mf(R, 0)
    
    # run the simulations
    meas = zeros(length(θs), length(βs), R+1)
    nins = zeros(length(θs), length(βs))
    emax = zeros(length(θs), length(βs))
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j])
        dm.x[:] = dm.pc[:fxp]
        meas[i,j,:] = abs.(eigvals(dm))
        nins[i,j] = sum(meas[i,j,:] .> 1.00001)
        emax[i,j] = maximum(meas[i,j,:])
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, nins')
    cbar = Colorbar(f[1,2], hm, label="Unstable modes")
    ax.title = L"Refractive model$$"
    save(plotsdir("refractive-unstable-plane.pdf"), f)
    display(f)
end

# Maximum eigenvalue in β-θ plane
begin
    J = 1
    R = 3
    θs = range(-1, 1, 101)
    βs = range(0, 20, 101)
    ic = spike_ic_mf(R, 0)
    
    # run the simulations
    meas = zeros(length(θs), length(βs), R+1)
    nins = zeros(length(θs), length(βs))
    emax = zeros(length(θs), length(βs))
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j])
        dm.x[:] = dm.pc[:fxp]
        meas[i,j,:] = abs.(eigvals(dm))
        nins[i,j] = sum(meas[i,j,:] .> 1.00001)
        emax[i,j] = maximum(meas[i,j,:])
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, emax')
    cbar = Colorbar(f[1,2], hm, 
        label="Maximum eigenvalue")
    ax.title = "Refractive model"
    #save(plotsdir("stability-line.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Integrator model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

# Visualize fixed point
begin
    J = 0.1
    θ = 1
    β = 50
    Q = 100
    α = 0.1
    I = 0.1
    C = exponential_weights(Q, α)
    fxp = integrator_fxp(J, θ, β, I, C)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 1:Q+1, fxp)
    #xlims!(ax, 1, 20)
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; Q = %$Q; \alpha=%$α, I = %$I$"
    display(f)
end

# Eigenvalues along J
begin
    Js = range(-4, 4, 51)
    θ = 1
    β = 10
    I = 0.1
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(0, Q)
    
    # run the simulations
    meas = zeros(length(Js), Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([Js]))    
        dm = IntegratorIMF(ic, Js[i], θ, β, I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = abs.(eigvals(dm))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"J", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, Js, meas[:,i])
    end
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β; Q = %$Q; \alpha=%$α$"
    save(plotsdir("integrator-stability-J.pdf"), f)
    display(f)
end

# Eigenvalues along α
begin
    J = 0.2
    θ = 1
    β = 20
    I = 0.1
    αs = range(0, 0.2,51)
    Q = 100
    ic = spike_ic_mf(0, Q)
    
    # run the simulations
    meas = zeros(length(αs), Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([αs]))    
        C = exponential_weights(Q, αs[i])
        dm = IntegratorIMF(ic, J, θ, β, I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = abs.(eigvals(dm))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\alpha", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, αs, meas[:,i])
    end
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β; Q = %$Q$"
    save(plotsdir("integrator-stability-alpha.pdf"), f)
    display(f)
end

# Eigenvalues along β
begin
    J = 0.3
    Q = 100
    θ = 1
    I = 0.2
    βs = range(0, 50, 51)
    ic = spike_ic_mf(0, Q)

    #C = ones(Q)
    α = 0.2
    C = exponential_weights(Q, α)
    
    # run the simulations
    meas = zeros(length(βs), Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = IntegratorIMF(ic, J, θ, βs[i], I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = abs.(eigvals(dm))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\beta", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, βs, meas[:,i])
    end
    #ylims!(ax, 0.9, 1.1)
    ax.title = L"Integrator model, $\theta = %$θ; Q = %$Q; \alpha=%$α$"
    save(plotsdir("integrator-stability-beta.pdf"), f)
    display(f)
end

# Eigenvalues along θ
begin
    J = 0.1
    θs = range(0, 2, 21)
    β = 40
    I = 0.1
    α = 0.1
    Q = 50
    ic = spike_ic_mf(0, Q)
    C = exponential_weights(Q, α)
    
    # run the simulations
    meas = zeros(length(θs), Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = IntegratorIMF(ic, J, θs[i], β, I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = abs.(eigvals(dm))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\theta", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, θs, meas[:,i])
    end
    ylims!(ax, 0.9, 1.1)
    ax.title = L"Integrator model, $\beta = %$β; Q = %$Q; \alpha=%$α$"
    save(plotsdir("integrator-stability-theta.pdf"), f)
    display(f)

end

# Eigenvalues in β-θ plane
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
        
        try
            dm = IntegratorIMF(ic, J, θs[i], βs[j], I, C)
            dm.x[:] = dm.pc[:fxp]
            meas[i,j,:] = abs.(eigvals(dm))
            nins[i,j] = sum(meas[i,j,:] .> 1.00001)
            emax[i,j] = maximum(meas[i,j,2:end])
        catch
            println("Error at θ = ", θs[i], " β = ", βs[j])
        end
    end
end
begin
    # plot
    f = Figure()
    ax = Axis(f[1,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, nins')
    cbar = Colorbar(f[1,2], hm, label="Unstable modes")
    ax.title = L"Integrator model"
    save(plotsdir("integrator-unstable-plane.pdf"), f)
    display(f)
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Complete model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Visualize fixed point
begin
    J = 1
    θ = 0
    β = 1
    R = 3
    Q = 5
    α = 0.1
    C = exponential_weights(Q, α)
    fxp = complete_ising_fxp(J, θ, β, R, Q, α)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 1:R+Q+1, fxp)
    #xlims!(ax, 1, 20)
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; R = %$R; Q = %$Q; \alpha=%$α$"
    display(f)
end