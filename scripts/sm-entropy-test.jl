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

begin
    N = 10
    J = 0.1
    θ = 1
    β = 20
    I = 0.1
    R = 0
    Q = 50
    α = 0.1
    C = exponential_weights(Q, α)
    n = random_ic_sm(N,R+Q)
    sm = CombinedSM(J, θ, β, I, n, R, C) 
    #sm = RefractiveSM(J, θ, β, I, n, R)
    forward!(sm, 2000)
    fdist = fdist_traj!(sm, 100)
end

#S, fdist = entropy!(sm, 2000)
fdist = fdist_traj!(sm, 2000)

# begin
#     fig = Figure()
#     ax = Axis(fig[1, 1], title="Entropy", ylabel = L"S/N")
#     lines!(ax, S[:,1], color = :blue, label = L"S")
#     lines!(ax, S[:,2], color = :red, label = L"S_r")
#     axislegend(ax, position = :rt, title = "Legend")

#     ax = Axis(fig[2, 1], ylabel = L"\sigma_t/N", xlabel = L"t")
#     lines!(ax, S[:,2]-S[:,1], color = :blue, label = L"")

#     ax = Axis(fig[3, 1], ylabel = L"n", xlabel = L"t")
#     lines!(ax, fdist[:,1,1], color = :blue, label = L"p(n)")

#     display(fig)
# end

# fdist = fdist_traj!(sm, 100)

# i = 1
# Nf = fdist[i, 1, :]
# Nb = fdist[i+2, 2, :] 

# hf = [sm.β*(local_current(sm.J,τ,N2a(Nf,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]
# hb = [sm.β*(local_current(sm.J,τ,N2a(Nb,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]
# hf = [hf[1:sm.Q]..., hf[sm.Q]]
# hb = [hb[1:sm.Q]..., hb[sm.Q]]

# NfR = Nf[sm.R+1:end]/sum(Nf[sm.R+1:end])
# NbR = Nb[sm.R+1:end]/sum(Nb[sm.R+1:end])

# S[i,1] = NfR'*(-hf.*tanh.(hf).+log.(2 .*cosh.(hf)))

# hf2 = hf*ones(1,Ncap(sm)-R)
# hb2 = ones(Ncap(sm)-R,1)*hb'
# Sr2 = (-hb2.*tanh.(hf2).+log.(2 .*cosh.(hb2)))

# Sr2

# lines(Nb[sm.R+1:end])


# S[i,2] = NfR'*Sr2*NbR

# hf = [sm.β*(local_current(sm.J,τ,N2a(Nf,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]
# hb = [sm.β*(local_current(sm.J,τ,N2a(Nb,sm.Q),sm.C,sm.I)-sm.θ) for τ in 1:sm.Q]



begin
    idx = 93
    fig = Figure()

    Nf, Nb = fdist[idx, 1, :], fdist[idx, 2, :]
    af = fdist[idx:idx+Ncap(sm)-1,2,1]

    ax = Axis(fig[1, 1])
    lines!(ax, 0:Ncap(sm)-1, Nf, color = :blue, label=L"p(n)")
    lines!(ax, 0:Ncap(sm)-1, Nb, color = :red, label= L"p(\hat{n})")
    axislegend(ax, position = :rt, title = "Legend")

    ax = Axis(fig[2, 1], ylabel = L"n", xlabel = L"t")
    lines!(ax, 0:Ncap(sm)-1, af, color = :green, label=L"a_{t+τ}")
    lines!(ax, 0:Ncap(sm)-1, Nb, color = :red, label= L"p(\hat{n})")

    test = fdist[idx:idx+Ncap(sm)-1,2,1]
    test1 = zeros(Ncap(sm))
    for i in 1:Ncap(sm)
        test1[i] = min(test[i], (1-sum(test1)))
    end

    test2 = zeros(Ncap(sm))
    
    for i in 1:Ncap(sm)
        x = Nf[i:end]
        currs = [(sm.C[1:min(τ,length(x))]'*(sm.J*x[1:min(τ,length(x))] .+ sm.I)) for τ in 1:sm.Q]
        fps = 0.5.+0.5*tanh.(sm.β*(currs .- sm.θ))
        fps = [fps[1:sm.Q]..., fps[sm.Q]]
        println(length(fps), " ", length(x))
        test2[i] = fps[1]
        #
    end
    
    #lines!(ax, 0:Ncap(sm)-1, test, color = :black, label=L"p(\hat{n})_{test}")
    lines!(ax, 0:Ncap(sm)-1, test1, color = :black, label=L"p(\hat{n})_{t1}")
    lines!(ax, 0:Ncap(sm)-1, test2, color = :orange, label=L"p(\hat{n})_{t2}")
    axislegend(ax, position = :rt, title = "Legend")
    println("sum test1: ", sum(test1))
    println("sum test2: ", sum(test2))

    display(fig)
end

# entropy along β
begin
    N = 1000
    J = 1
    θ = 0
    βs = range(0, 10, 11)
    I = 0
    R = 3
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    ic = random_ic_sm(N, R+Q)

    nequi, nmeas = 2000, 2000
    
    # run the simulations
    meas = zeros(length(βs), 2)
    #Threads.@threads 
    for (i,) in ProgressBar(idx_combinations([βs]))
        sm = CombinedSM(J, θ, βs[i], I, ic, R, C)
        sm = RefractiveSM(J, θ, βs[i], I, ic, R)
        forward!(sm, nequi)
        S = entropy!(sm, nmeas)
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