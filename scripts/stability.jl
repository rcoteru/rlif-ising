using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));   
    using ProgressBars
    using CairoMakie   
    CairoMakie.activate!()
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Refractive model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Visualize fixed point
begin
    J = 1
    θ = 0
    β = 5
    R = 3
    I = 0
    fxp = refractive_fxp(J, θ, β, I, R)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 1:R+1, fxp)
    #xlims!(ax, 1, 20)
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# Eigenvalues along β
begin 
    J = 1
    R = 3
    θ = 0
    βs = range(0, 50, 101)
    I = 0
    ic = spike_ic_mf(R, 0)
    
    # run the simulations
    meas = zeros(length(βs), R+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = RefractiveIMF(ic, J, θ, βs[i], I)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"λ")
    for i in 1:R+1
        lines!(ax, βs, meas[:,i], color=:black, linewidth=1.5)
    end
    # set title with θ dynamically
    ax.title = L"Refractive model, $\theta = %$θ, R = %$R$"
    save(plotsdir("refractive-stability-beta.pdf"), f)
    display(f)
end

# Eigenvalues along θ
begin 
    J = 1
    R = 3
    θs = range(-1, 1, 101)
    β = 10
    I = 0
    ic = spike_ic_mf(R, 0)
    
    # run the simulations
    meas = zeros(length(θs), R+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([θs]))    
        dm = RefractiveIMF(ic, J, θs[i], β, I)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\theta", ylabel=L"\lambda")
    for i in 1:R+1
        lines!(ax, θs, meas[:,i], color=:black, linewidth=1.5)
    end
    ylims!(ax, 0.8, 1.2)
    ax.title = L"Refractive model, $\beta = %$β, R = %$R$"
    save(plotsdir("refractive-stability-theta.pdf"), f)
    display(f)
end

# Unstable eigenvalues in β-θ plane
begin
    J = 1
    R = 3
    θs = range(-1, 1, 101)
    βs = range(0, 60, 101)
    I = 0
    ic = spike_ic_mf(R, 0)
    tol = 1e-8
    
    # run the simulations
    meas = zeros(length(θs), length(βs), R+1)
    nins = zeros(length(θs), length(βs))
    emax = zeros(length(θs), length(βs))
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j], I)
        dm.x[:] = dm.pc[:fxp]
        meas[i,j,:] = abs.(eigvals(dm))
        nins[i,j] = sum(meas[i,j,:] .> 1+tol)
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
    I = 0
    ic = spike_ic_mf(R, 0)
    tol = 1e-8
    
    # run the simulations
    meas = zeros(length(θs), length(βs), R+1)
    nins = zeros(length(θs), length(βs))
    emax = zeros(length(θs), length(βs))
    Threads.@threads for (i,j) in ProgressBar(idx_combinations([θs, βs]))    
        dm = RefractiveIMF(ic, J, θs[i], βs[j], I)
        dm.x[:] = dm.pc[:fxp]
        meas[i,j,:] = abs.(eigvals(dm))
        nins[i,j] = sum(meas[i,j,:] .> 1+tol)
        emax[i,j] = maximum(meas[i,j,:])
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], ylabel=L"\theta", xlabel=L"\beta")
    hm = heatmap!(ax, βs, θs, emax', colorrange=(1, 1.2))
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
    lines!(ax, 0:Q-1, C)
    ax.title = L"Exponential weights, $\alpha=%$α$"
    xlims!(ax, 0, Q-1)
    ylims!(ax, 0, 1)
    display(f)
end

# Visualize fixed point
begin
    J = 0.1
    θ = 1
    β = 21
    Q = 100
    α = 0.1
    I = 0.1
    C = exponential_weights(Q, α)
    fxp = integrator_fxp(J, θ, β, I, C)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 0:Q, fxp)
    xlims!(ax, 0, Q)
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; Q = %$Q; \alpha=%$α, I = %$I$"
    display(f)
end

# Visualize eigenvectors
begin
    J = 0.1
    θ = 1
    β = 17.5
    Q = 100
    α = 0.1
    I = 0.1
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(0, Q)
    
    # run the simulations
    dm = IntegratorIMF(ic, J, θ, β, I, C)
    dm.x[:] = dm.pc[:fxp]

    vals = eigvals(dm)
    vecs = eigvecs(dm)
    # sort eigenvalues and eigenvectors
    idx = sortperm(abs.(vals), rev=true)
    vals = vals[idx]
    vecs = vecs[:,idx]
    # get the largest eigenvalue and its eigenvector
    λ = vals[1]
    v = vecs[:,1]
    # sets preferential sign of eigenvector
    if real(v[1]) < 0
        v = -v
    end
    
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"v_n")
    lines!(ax, 0:Q, real(v), color=:black, linewidth=1.5)
    lines!(ax, 0:Q, imag(v), color=:red, linewidth=1.5)
    ax.title = L"Eigenvector of largest eigenvalue $|\lambda| = %$(abs(λ))$"
    
    println("Eigenvalue: ", abs(λ))
    
    display(f)
end

# Eigenvalues along J
begin
    Js = range(-0.5,1.5, 51)
    θ = 1
    β = 20
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
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"J", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, Js, meas[:,i], color=:black, linewidth=1.5)
    end
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β; Q = %$Q; \alpha=%$α$"
    save(plotsdir("integrator-stability-J.pdf"), f)
    display(f)
end

# Eigenvalues along α
begin
    J = 0.2
    θ = 1
    β = 30
    I = 0.1
    αs = range(0, 0.15, 51)
    Q = 100
    ic = spike_ic_mf(0, Q)
    
    # run the simulations
    meas = zeros(length(αs), Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([αs]))    
        C = exponential_weights(Q, αs[i])
        dm = IntegratorIMF(ic, J, θ, β, I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\alpha", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, αs, meas[:,i], color=:black, linewidth=1.5)
    end
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β; Q = %$Q$"
    save(plotsdir("integrator-stability-alpha.pdf"), f)
    display(f)
end

# Eigenvalues along β
begin
    J = 0.2
    θ = 1
    I = 0.1
    βs = range(0, 20, 51)
    Q = 50
    
    #C = ones(Q)
    α = 0.1
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(0, Q)
    
    # run the simulations
    meas = zeros(length(βs), Q+1)
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
        dm = IntegratorIMF(ic, J, θ, βs[i], I, C)
        dm.x[:] = dm.pc[:fxp]
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\beta", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, βs, meas[:,i], color=:black, linewidth=1.5)
    end
    ylims!(ax, 0.9, 1.1)
    ax.title = L"Integrator model, $\theta = %$θ; Q = %$Q; \alpha=%$α$"
    save(plotsdir("integrator-stability-beta.pdf"), f)
    display(f)
end

# Eigenvalues along θ
begin
    J = 0.1
    θs = range(0, 2, 21)
    β = 20
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
        meas[i,:] = sort(abs.(eigvals(dm)))
    end

    # plot
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"\theta", ylabel=L"\lambda")
    for i in 1:Q+1
        lines!(ax, θs, meas[:,i], color=:black, linewidth=1.5)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Combined model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Visualize fixed point
begin
    J = 0.1
    θ = 1
    β = 5
    I = 0.1
    R = 5
    Q = 50
    α = 0.1
    C = exponential_weights(Q, α)
    fxp = combined_fxp(J, θ, β, I, R, C)
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"P(n)")
    lines!(ax, 1:R+Q+1, fxp)
    #xlims!(ax, 1, 20)
    ax.title = L"Fixed point, $\theta = %$θ; \beta = %$β; R = %$R; Q = %$Q; \alpha=%$α$"
    display(f)
end

# Visualize eigenvectors
begin
    J = 0.1
    θ = 1
    β = 13
    I = 0.1
    R = 10
    Q = 50
    α = 0.1
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(R, Q)
    
    # run the simulations
    dm = CombinedIMF(ic, J, θ, β, I, C)
    dm.x[:] = dm.pc[:fxp]

    vals = eigvals(dm)
    vecs = eigvecs(dm)
    # sort eigenvalues and eigenvectors
    idx = sortperm(abs.(vals), rev=true)
    vals = vals[idx]
    vecs = vecs[:,idx]
    # get the largest eigenvalue and its eigenvector
    λ = vals[1]
    v = vecs[:,1]
    # sets preferential sign of eigenvector
    if real(v[1]) < 0
        v = -v
    end

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"n", ylabel=L"v_n")
    lines!(ax, 1:R+Q+1, real(v), color=:black, linewidth=1.5)
    lines!(ax, 1:R+Q+1, imag(v), color=:red, linewidth=1.5)
    ax.title = L"Eigenvector of largest eigenvalue $|\lambda| = %$(abs(λ))$"
    println("Eigenvalue: ", abs(λ))
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
    save(plotsdir("combined-stability-beta.pdf"), f)
    display(f)
end

# Transition β as a function of R
begin
    J = 0.2
    θ = 1
    Rs = range(1, 20)
    I = 0.1
    Q = 50
    α = 0.1
    βs = range(0, 20, 51)
    C = exponential_weights(Q, α)
    tol = 1e-8
    βtrans = zeros(length(Rs))
    for R in Rs
        println("R = ", R)
        ic = spike_ic_mf(R, Q)
        meas = zeros(length(βs), R+Q+1)
        Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))    
            dm = CombinedIMF(ic, J, θ, βs[i], I, C)
            dm.x[:] = dm.pc[:fxp]
            meas[i,:] = sort(abs.(eigvals(dm)))
        end
        idx = findfirst(meas[:,Q+R+1].>(1+tol))
        βtrans[R] = βs[idx]
    end
end
begin
    # plot
    f = Figure()
    ax = Axis(f[1,1], title=L"Combined model: $\beta_T$ vs $R$",
        xlabel=L"R", ylabel=L"\beta_T")
    lines!(Rs, βtrans, color=:black, linewidth=1.5)
    display(f)
end