using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));  
    using ProgressMeter
    using CairoMakie
    CairoMakie.activate!()
    using FFTW
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Refractive model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Mean field time series
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# MF: Time series 
begin
    J = 1
    θ = 0
    I = 0
    β = 30
    R = 3

    nequi, nmeas = 1000, 2000

    #ic = spike_ic_mf(R, 0)
    ic = random_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β, I)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, 200
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready"]
    for i in [1]
        lines!(ax, t0:tf, traj[t0:tf,i], label=labels[i])
    end
    axislegend(ax, position = :lt)

    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R; J = %$J$"
    display(f)
end

# MF: Envelope
begin
    J = 1
    θ = 0
    I = 0
    β = 30
    R = 16

    nequi = 400000
    nmeas = 10000

    #ic = spike_ic_mf(R, 0)
    ic = random_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β,I)

    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, 10000
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready"]
    for i in [1]
        lines!(ax, t0:R+1:tf, traj[t0:R+1:tf,i], label=labels[i])
    end
    axislegend(ax, position = :lt)

    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)

end

# MF: Time Series FFT
begin
    J = 1
    θ = 0
    β = 10
    I = 0
    R = 3

    nequi = 1000
    nmeas = 5000

    ic = random_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β,I)

    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:3
        x = rfftfreq(length(traj[t0:tf,i]))[2:end]
        y = abs.(rfft(traj[t0:tf,i]))[2:end]
        lines!(ax, x, y, label=labels[i])
    end

    # plot harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    display(f)
end

# MF: Envelope FFT
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    I = 1

    nequi = 1000
    nmeas = 5000

    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:3
        x = rfftfreq(length(traj[t0:R+1:tf,i]))[2:end]
        y = abs.(rfft(traj[t0:R+1:tf,i]))[2:end]
        lines!(ax, x, y, label=labels[i])
    end

    # plot harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    display(f)
end

# MF: Time Series Poincare section
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    I = 0

    nequi = 1000
    nmeas = 1000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β, I)
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a(t)", ylabel=L"a(t+1)")
    scatter!(ax, traj[1:end-1,1], traj[2:end,1], markersize = 2, color = :black)
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# MF: Envelope Poincare section
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    I = 0

    nequi = 1000
    nmeas = 1000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β,I)
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a(t)", ylabel=L"a(t+R+1)")
    scatter!(ax, traj[1:end-(R+1),1], traj[R+2:end,1], markersize = 2, color = :black)
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# Spin model time series
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# SM: Time series
begin
    J = -1
    θ = 0
    β = 10
    R = 3
    I = 0.1
    N = 200

    nequi = 1000
    nmeas = 2000

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, I, ic, R)
    
    forward!(sm, nequi)
    traj = network_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, 500
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready", "Fxp", "Kuramoto"]
    colors = [:black, :red, :blue, :green, :orange]
    for i in [1,4,5]
        lines!(ax, t0:tf, traj[t0:tf,i], label=labels[i], color = colors[i])
    end

    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# SM: Envelope
begin
    J = -1
    θ = 0
    β = 10
    R = 3
    N = 200

    nequi = 1000
    nmeas = 1000

    forward!(sm, nequi)
    traj = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, 400
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    lines!(ax, t0:R+1:tf, traj[t0:R+1:tf,1], label="Firing")
    lines!(ax, t0:R+1:tf, traj[t0:R+1:tf,2], label="Refractive")
    lines!(ax, t0:R+1:tf, traj[t0:R+1:tf,3], label="Ready")
    axislegend(ax, position = :lt)

    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# SM: Time Series FFT
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 200

    nequi = 1000
    nmeas = 1000

    forward!(sm, nequi)
    traj = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:3
        x = rfftfreq(length(traj[t0:tf,i]))[2:end]
        y = abs.(rfft(traj[t0:tf,i]))[2:end]
        lines!(ax, x, y, label=labels[i])
    end

    # plot harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    display(f)
end

# SM: Envelope FFT
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 200

    nequi = 1000
    nmeas = 1000

    forward!(sm, nequi)
    traj = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:3
        x = rfftfreq(length(traj[t0:R+1:tf,i]))[2:end]
        y = abs.(rfft(traj[t0:R+1:tf,i]))[2:end]
        lines!(ax, x, y, label=labels[i])
    end

    # plot harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    display(f)
end

# SM: Time Series Poincare section
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj = adv_traj!(sm, nmeas, parallel=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a_t", ylabel=L"a_{t+1}")
    scatter!(ax, traj[1:end-1,1], traj[2:end,1], markersize = 2, color = :black)
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# SM: Envelope Poincare section
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj = adv_traj!(sm, nmeas, parallel=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a_t", ylabel=L"a_{t+R+1}")
    scatter!(ax, traj[1:end-(R+1),1], traj[R+2:end,1], markersize = 2, color = :black)
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    display(f)
end

# Avalanches
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# SM: Avalanches
begin
    J = 1
    θ = -0.167
    β = 12
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 1000

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    
    forward!(sm, nequi)
    traj = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, 1000
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready"]
    for i in [1]
        lines!(ax, t0:tf, traj[t0:tf,i], label=labels[i])
    end
    axislegend(ax, position = :lt)

    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    save(plotsdir("refractive-sm-avalanches.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# MF vs SM: Time series
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β)
    forward!(dm, nequi)
    traj_mf = trajectory!(dm, nmeas, ft=true)

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj_sm = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, 30
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:1
        lines!(ax, t0:tf, traj_mf[t0:tf,i], label="MF: "*labels[i], color = :red)
        lines!(ax, t0:tf, traj_sm[t0:tf,i], label="SM: "*labels[i], color = :blue)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    axislegend(ax, position = :lt)
    save(plotsdir("refractive_mf_vs_sm_ts.pdf"), f)
    display(f)
end

# MF vs SM: Time series FFT
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β)
    forward!(dm, nequi)
    traj_mf = trajectory!(dm, nmeas, ft=true)

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj_sm = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:1
        x = rfftfreq(length(traj_mf[t0:tf,i]))[2:end]
        y_mf = abs.(rfft(traj_mf[t0:tf,i]))[2:end]
        y_sm = abs.(rfft(traj_sm[t0:tf,i]))[2:end]
        lines!(ax, x, y_mf, label="MF: "*labels[i], color = :red)
        lines!(ax, x, y_sm, label="SM: "*labels[i], color = :blue)
    end

    # plot harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    save(plotsdir("refractive_mf_vs_sm_ts-fft.pdf"), f)
    display(f)
end

# MF vs SM: Envelope
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β)
    forward!(dm, nequi)
    traj_mf = trajectory!(dm, nmeas, ft=true)

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj_sm = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, 500
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:1
        lines!(ax, t0:R+1:tf, traj_mf[t0:R+1:tf,i], label="MF: "*labels[i], color = :red)
        lines!(ax, t0:R+1:tf, traj_sm[t0:R+1:tf,i], label="SM: "*labels[i], color = :blue)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    axislegend(ax, position = :lt)
    save(plotsdir("refractive_mf_vs_sm_ts-envelope.pdf"), f)
    display(f)
end

# MF vs SM: Envelope FFT Comparison
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 3000
    nmeas = 10000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β)
    forward!(dm, nequi)
    traj_mf = trajectory!(dm, nmeas, ft=true)

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj_sm = adv_traj!(sm, nmeas, parallel=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:1
        x = rfftfreq(length(traj_mf[t0:R+1:tf,i]))[2:end]
        y_mf = abs.(rfft(traj_mf[t0:R+1:tf,i]))[2:end]
        y_sm = abs.(rfft(traj_sm[t0:R+1:tf,i]))[2:end]
        lines!(ax, x, y_mf, label="MF: "*labels[i], color = :red)
        lines!(ax, x, y_sm, label="SM: "*labels[i], color = :blue)
    end

    # plot harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    save(plotsdir("refractive_mf_vs_sm_ts-envelope-fft.pdf"), f)
    display(f)
end

# MF vs SM: Time Series Poincare section
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β)
    forward!(dm, nequi)
    traj_mf = trajectory!(dm, nmeas, ft=true)

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj_sm = adv_traj!(sm, nmeas, parallel=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a(t)", ylabel=L"a(t+1)")
    scatter!(ax, traj_sm[1:end-1,1], traj_sm[2:end,1], markersize = 3, color = :blue)
    scatter!(ax, traj_mf[1:end-1,1], traj_mf[2:end,1], markersize = 3, color = :red)
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    save(plotsdir("refractive_mf_vs_sm_ps.pdf"), f)
    display(f)
end

# MF vs SM: Envelope Poincare section
begin
    J = 1
    θ = 0
    β = 10
    R = 3
    N = 1000

    nequi = 1000
    nmeas = 5000

    ic = spike_ic_mf(R, 0)
    dm = RefractiveIMF(ic, J, θ, β)
    forward!(dm, nequi)
    traj_mf = trajectory!(dm, nmeas, ft=true)

    ic = spike_ic_sm(N)
    sm = RefractiveSM(J, θ, β, ic, R)
    forward!(sm, nequi)
    traj_sm = adv_traj!(sm, nmeas, parallel=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a(t)", ylabel=L"a(t+R+1)")
    scatter!(ax, traj_sm[1:end-(R+1),1], traj_sm[R+2:end,1], markersize = 3, color = :blue)
    scatter!(ax, traj_mf[1:end-(R+1),1], traj_mf[R+2:end,1], markersize = 3, color = :red)
    ax.title = L"Refractive model, $\theta = %$θ; \beta = %$β; R = %$R$"
    save(plotsdir("refractive_mf_vs_sm_ps-envelope.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Integrator Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# MF: Time series 
begin
    J = 0.1
    θ = 1
    β = 20
    I = 0.1
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    # ic = spike_ic_mf(0, Q)
    

    nequi, nmeas = 4000, 2000

    #ic = spike_ic_mf(0, Q)
    ic= random_ic_mf(0, Q)
    dm = IntegratorIMF(ic, J, θ, β, I, C)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, 500
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    for i in [1]
        lines!(ax, t0:tf, traj[t0:tf,i], label=labels[i])
    end
    axislegend(ax, position = :lt)
    display(f)
end


# MF: Time series FFT
begin
    J = 0.16
    θ = 1
    β = 12
    I = 0.1
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(0, Q)

    nequi, nmeas = 10000,1000

    ic = spike_ic_mf(0, Q)
    dm = IntegratorIMF(ic, J, θ, β, I, C)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:3
        x = rfftfreq(length(traj[t0:tf,i]))[2:end]
        y = abs.(rfft(traj[t0:tf,i]))[2:end]
        lines!(ax, x, y, label=labels[i])
    end

    # decorations
    ax.title = L"Integrator model, $\theta = %$θ; \beta = %$β; J = %$J$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    display(f)
end

# MF: Time series Poincare section
begin
    J = 0.2
    θ = 1
    β = 12
    I = 0.1
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    ic = spike_ic_mf(0, Q)

    nequi, nmeas = 10000,1000

    ic = spike_ic_mf(0, Q)
    dm = IntegratorIMF(ic, J, θ, β, I, C)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"a(t)", ylabel=L"a(t+1)")
    lines!(ax, traj[1:end-1,1], traj[2:end,1], color = :black)
    display(f)
end

# Spin model time series
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# SM: Time series, activity, fields
begin
    N = 500
    J = 0.2
    β = 15
    θ = 1
    I = 0.1
    α = 0.1
    Q = 50
    C = exponential_weights(Q, α)
    #n = spike_ic_sm(N)
    n = random_ic_sm(N, Q)
    sm = IntegratorSM(J, θ, β, I, n, C)

    nequi, nmeas = 1000, 500
    forward!(sm, nequi)
    sh, lcs = spinwise_traj!(sm, nmeas, parallel=true)

    f = Figure()
    ax1 = Axis(f[1,1], ylabel=L"a(t)")
    lines!(ax1, 1:nmeas, (mean(sh, dims=2)[:].+1)./2)
    xlims!(ax1, 0,nmeas)
    hidexdecorations!(ax1)

    ax2 = Axis(f[2,1], ylabel=L"neuron $i$")
    heatmap!(ax2, sh)
    hidexdecorations!(ax2)

    ax3 = Axis(f[3,1], xlabel=L"t", ylabel=L"h_i(t)")
    for i in 1:N
        lines!(ax3, lcs[:,i])
    end
    xlims!(ax3, 0, nmeas)
    ax1.title = L"Integrator model, $N = %$N; Q = %$Q; J = %$J; θ = %$θ; β = %$β; I = %$I; α = %$α$"
    #save(plotsdir("integrator-sm-ts.pdf"), f)
    display(f)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Combined Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# MF: Time series
begin
    J = 0.2
    θ = 1
    β = 20
    I = 0.1
    R = 3
    Q = 50
    C = exponential_weights(Q, 0.1)

    nequi, nmeas = 10000, 1000

    ic = random_ic_mf(R, Q)
    dm = CombinedIMF(ic, J, θ, β, I, C)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1,200
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"t", ylabel=L"n")
    labels = ["Firing", "Refractive", "Ready", "Fxp", "rK", "phiK"]
    colors = [:black, :red, :blue, :green, :orange, :purple]
    for i in [1]
        lines!(ax, t0:tf, traj[t0:tf,i], label=labels[i], color = colors[i])
    end
    axislegend(ax, position = :lt)

    ax.title = L"Combined Model, $\theta = %$θ; \beta = %$β; R = %$R; J = %$J$"
    display(f)
end

# MF: Time series FFT
begin
    J = 0.1
    θ = 0.1
    R = 5
    β = 20
    I = 0.1 
    Q = 50
    C = exponential_weights(Q, 0.1)

    nequi, nmeas = 10000, 1000

    ic = random_ic_mf(R, Q)
    dm = CombinedIMF(ic, J, θ, β, I, C)
    
    forward!(dm, nequi)
    traj = trajectory!(dm, nmeas, ft=true)

    t0, tf = 1, nmeas
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"f", ylabel=L"|n(f)|")
    labels = ["Firing", "Refractive", "Ready"]
    for i in 1:3
        x = rfftfreq(length(traj[t0:tf,i]))[2:end]
        y = abs.(rfft(traj[t0:tf,i]))[2:end]
        lines!(ax, x, y, label=labels[i])
    end

    # harmonics
    harmonics = [1, 2, 3]
    for h in harmonics
        vlines!(ax, 1/(h*(R+1)), [0, 1], color = :black, linestyle = :dash)
    end

    # decorations
    ax.title = L"Combined Model, $\theta = %$θ; \beta = %$β; R = %$R; J = %$J$"
    ax.yticklabelsvisible = false
    axislegend(ax)
    display(f)
end