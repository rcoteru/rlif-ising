using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));   
    using ProgressBars
    using CairoMakie
end

begin
    N = 2000
    J = 0.2
    θ = 1
    β = 30
    R = 3
    Q = 50
    I = 0.1
    α = 0.1
    C = exponential_weights(Q, α)
    n = random_ic_sm(N,R+Q)
    sm = CombinedSM(J, θ, β, I, n, R, C) 
    forward!(sm, 100)
    fdist = fdist_traj!(sm, 100)
end

begin
    fig = Figure()
    idx = 50
    ax = Axis(fig[1, 1])
    
    lines!(ax, 0:Ncap(sm)-1, fdist[idx,1,:], color = :blue, label=L"p(n)")
    lines!(ax, 0:Ncap(sm)-1, fdist[idx,2,:], color = :red, label= L"p(\hat{n})")
    axislegend(ax, position = :rt, title = "Legend")

    fig
end

# entropy along β
begin
    n = 500
    J = 0.1
    θ = 1
    βs = range(0, 50, 51)
    I = 0.1
    R = 3
    α = 0.1
    C = exponential_weights(Q, α)
    ic = random_ic_sm(N, R+Q)

    nequi, nmeas = 500, 2000
    
    # run the simulations
    meas = zeros(length(βs), 2)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([βs]))
        dm = CombinedSM(J, θ, βs[i], I, ic, R, C)
        forward!(dm, nequi)
        S = entropy!(dm, nmeas)
        meas[i,:] = mean(S, dims = 1)
    end
end
begin

    title = "J = $J, θ = $θ, I = $I, R = $R, Q = $Q, α = $α"

    # plot
    f = Figure()
    ax = Axis(f[1, 1], title=title, ylabel = L"S/N")
    lines!(ax, βs, meas[:,1], color = :blue, label = L"S")
    lines!(ax, βs, meas[:,2], color = :red, label = L"S_r")
    axislegend(ax, position = :rt, title = "Legend")


    ax = Axis(f[2, 1], ylabel = L"\sigma_t/N", xlabel = L"\beta")
    lines!(ax, βs, meas[:,2]-meas[:,1], color = :blue, label = L"")

    display(f)
end