using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));     
end

using CairoMakie

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Refractive model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

begin  

    # parameters
    N = 1000
    J = 1
    θ = 0
    β = 1
    I = 0.1
    R = 5

    # theoretical fixed points
    xt = refractive_fxp(J, θ, β, I, R)
    
    # mean field simulation
    ic = quiet_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β, I)
    forward!(dm, 1000)
    xm = dm.x

    # spin simulation
    n = quiet_ic_mf(R, 0)
    sm = RefractiveSM(J, θ, β, I, n, R)
    forward!(sm, 1000, parallel = true)
    xs = zeros(R+1)
    samps = 1000
    for i in 1:samps
        forward!(sm, 1, parallel = true)
        xs += n2N(sm)
    end 
    xs = xs./samps

    # plot
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, 1:R+1, xt, color = :black, linewidth = 2, 
        linestyle = :solid, label = "Theoretical")
    lines!(ax, 1:R+1, xm, color = :red, linewidth = 2, 
        linestyle = :dash, label = "Mean Field")
    lines!(ax, 1:R+1, xs, color = :blue, linewidth = 2, 
        linestyle = :dot, label = "Spin Model")
    axislegend(ax, position = :lt)
    ax.ylabel = L"P(n)"
    ax.xlabel = L"n"

    display(fig)
    save(plotsdir("refractive-fixed-points.pdf"), fig)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Integrator model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# THEO: Firing probability and transcendental
begin
    J = 0.1
    θ = 1
    β = 10
    Q = 50
    I = 0.1
    C = exponential_weights(Q, 0.1)

    # NOTE: for the integrator model
    # to have self-sustaining oscillations you need
    # to have non-zero probabilities for some tau at 
    # every value of a_0

    f = Figure()

    x = range(0, 1, length=200)

    ax = Axis(f[1,1:2])
    probs = stack([integrator_fxp_probs(i, J, θ, β, I, C)[1,:] for i in x])
    hm = heatmap!(ax, x, 1:Q, probs',colorrange=(0,1))
    Colorbar(f[1,3], hm, label = "Probability")
    
    ax = Axis(f[2,1:3])
    y = [integrator_fxp_transcendental(i, J, θ, β, I, C) for i in x]
    lines!(ax, x, y,color = :black, linewidth = 2, linestyle = :solid)
    hlines!(ax, [0], color = :black, linewidth = 1, linestyle = :dash)
    
    # ax = Axis(f[3,1:3])
    # currs = stack([integrator_fxp_currents(i, J, θ, β, I, C)[Q] for i in x])
    # y = [exp(-2*currs[i]) for i in 1:length(x)]
    # lines!(ax, x, y,color = :blue, linewidth = 2, linestyle = :solid)
    # y = [probs[Q,i] for i in 1:length(x)]
    # lines!(ax, x, y,color = :black, linewidth = 2, linestyle = :solid)
    # y = [x[i]*prod(1 .-probs[1:Q,i]) for i in 1:length(x)]
    # lines!(ax, x, y,color = :red, linewidth = 2, linestyle = :dash)
    # ylims!(ax, 0, 1)

    # print max and min
    Label(f[3,3], "Max: $(maximum(y))", fontsize = 12)
    Label(f[3,1], "Min: $(minimum(y))", fontsize = 12)

    display(f)

end

# THEO vs MF vs SM
begin

    # parameters
    N = 2000
    J = 0.1
    θ = 1
    β = 200
    Q = 50
    I = 0.1
    α = 0.1
    C = exponential_weights(Q, α)

    # theoretical fixed points
    xt = integrator_fxp(J, θ, β, I, C)

    # mean field simulation
    ic = random_ic_mf(Q, 0)
    dm = IntegratorIMF(ic, J, θ, β, I, C)
    forward!(dm, 2000)
    xm = dm.x

    # spin simulation
    #n = spike_ic_sm(N)
    n = random_ic_sm(N, Q)
    sm = IntegratorSM(J, θ, β, I, n, C)
    forward!(sm, 2000, parallel = true)
    xs = zeros(Q+1)
    samps = 1
    for i in 1:samps
        forward!(sm, 1, parallel = true)
        xs += n2N(sm)
    end
    xs = xs./samps

    # plot
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, 1:Q+1, xt, color = :black, linewidth = 2, 
        linestyle = :solid, label = "Theoretical")
    lines!(ax, 1:Q+1, xm, color = :red, linewidth = 2,
        linestyle = :dash, label = "Mean Field")
    lines!(ax, 1:Q+1, xs, color = :blue, linewidth = 2,
        linestyle = :dot, label = "Spin Model")
    axislegend(ax)  

    ax.ylabel = L"P(n)"
    ax.xlabel = L"n"

    display(fig)
    save(plotsdir("integrator-fixed-points.pdf"), fig)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Combined model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# THEO: Firing probability and transcendental
begin
    J = 0.1
    θ = 1
    β = 10
    I = 0.1
    R = 3
    Q = 50
    α = 0.1
    C = exponential_weights(Q, α)

    f = Figure()

    x = range(0, 1, length=200)

    ax = Axis(f[1,1:2])
    probs = stack([combined_fxp_currents(i,J,θ,β,I,R,C) for i in x])
    probs = 0.5.*(1 .+tanh.(probs))
    hm = heatmap!(ax, x, 1:Q, probs',colorrange=(0,1))
    Colorbar(f[1,3], hm, label = "Probability")
    
    ax = Axis(f[2,1:3])
    y = [combined_fxp_transcendental(i, J, θ, β, I, R, C) for i in x]
    lines!(ax, x, y,color = :black, linewidth = 2, linestyle = :solid)
    hlines!(ax, [0], color = :black, linewidth = 1, linestyle = :dash)
    
    # ax = Axis(f[3,1:3])
    # currs = stack([integrator_fxp_currents(i, J, θ, β, I, C)[Q] for i in x])
    # y = [exp(-2*currs[i]) for i in 1:length(x)]
    # lines!(ax, x, y,color = :blue, linewidth = 2, linestyle = :solid)
    # y = [probs[Q,i] for i in 1:length(x)]
    # lines!(ax, x, y,color = :black, linewidth = 2, linestyle = :solid)
    # y = [x[i]*prod(1 .-probs[1:Q,i]) for i in 1:length(x)]
    # lines!(ax, x, y,color = :red, linewidth = 2, linestyle = :dash)
    # ylims!(ax, 0, 1)

    # # print max and min
    # Label(f[4,3], "Max: $(maximum(y))", fontsize = 12)
    # Label(f[4,1], "Min: $(minimum(y))", fontsize = 12)

    display(f)
end

begin

    # parameters
    N = 2000
    J = 0.3
    θ = 1
    β = 10
    Q = 50
    I = 0.1
    α = 0.1
    C = exponential_weights(Q, α)
    R = 3

    # theoretical fixed points
    #xt = combined_fxp(J, θ, β, I, R, C)
    xt = zeros(R+Q+1)

    # mean field simulation
    ic = random_ic_mf(R, Q)
    dm = CombinedIMF(ic, J, θ, β, I, C)
    forward!(dm, 5000)
    xm = dm.x
    
    # spin simulation
    # n = spike_ic_sm(N)
    n = random_ic_sm(N, R+Q)
    sm = CombinedSM(J, θ, β, I, n, R, C)
    forward!(sm, 5000, parallel = true)
    xs = zeros(R+Q+1)
    samps = 1
    for i in 1:samps
        forward!(sm, 1, parallel = true)
        xs += n2N(sm)
    end
    xs = xs./samps

    # plot
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, 1:R+Q+1, xt, color = :black, linewidth = 2, 
        linestyle = :solid, label = "Theoretical")
    lines!(ax, 1:R+Q+1, xm, color = :red, linewidth = 2,
        linestyle = :dash, label = "Mean Field")
    lines!(ax, 1:R+Q+1, xs, color = :blue, linewidth = 2,
        linestyle = :dot, label = "Spin Model")
    axislegend(ax)  

    ax.ylabel = L"P(n)"
    ax.xlabel = L"n"

    display(fig)
    save(plotsdir("combined-fixed-points.pdf"), fig)
end