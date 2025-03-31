using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));  
    using ProgressBars
    using GLMakie
    GLMakie.activate!()
end

# animation: image of a spin model
# where the color represents the s.n variable

# Parameters
begin
    N = 900
    J = 0.2
    θ = 1
    β = 10
    I = 0.1
    R = 3
    Q = 50
    α = 0.1
    n = random_ic_sm(N, Q)
    C = exponential_weights(Q, α)
    sm = CombinedSM(J, θ, β, I, n, R, C)
    dm = CombinedIMF(n2N(sm), J, θ, β, I, C)

    tmax = 300
    begin
        img = Observable(n2img(sm, false))
        mf_traj = Observable([dm.x[1]])
        sm_traj = Observable([s2a(sm.s)])
        mf_dist = Observable(Vector(dm.x))
        sm_dist = Observable(n2N(sm))

        phase = combined_map_phase(dm.x, dm.p)
        nphase = kuramoto_phases(sm)
        mf_phase = Observable(combined_map_phase(dm.x, dm.p))
        sm_phase = Observable(nphase)

        mf_ikur = dm.x'*exp.(im.*phase)
        mf_kur = Observable([(real(mf_ikur), imag(mf_ikur))])
        sm_ikur = kuramoto(sm)
        sm_kur = Observable([(real(sm_ikur), imag(sm_ikur))])

        sw_phases = [exp(im*nphase[min(n+1, Q)]) for n in sm.n]
        sm_swkur = Observable([Point2f(real(sp), imag(sp)) for sp in sw_phases])

        fig = Figure(size = (1200, 400))

        param_string = "N=$N, J = $J, θ = $θ, β = $β, I = $I, R = $R, Q = $Q, α = $α"
        Label(fig[1, 1:4], L"RLIF Ising Model \\ $%$param_string$", 
        fontsize=16)

        ax = Axis(fig[2:3, 1], title=L"a_t", xlabel=L"t")
        lines!(ax, mf_traj, color = :red, label="mean-field")
        lines!(ax, sm_traj, color = :blue, label="spin-model")
        axislegend(ax, position = :rt)
        xlims!(ax, 0, tmax)
        ylims!(ax, 0, 1)

        ax = Axis(fig[2, 2], title=L"p_t(n)", xlabel=L"n")
        lines!(ax, 0:Ncap(sm)-1, mf_dist, color = :red)
        lines!(ax, 0:Ncap(sm)-1, sm_dist, color = :blue)
        xlims!(ax, 0,Ncap(sm)-1)
        ylims!(ax, 0, 1)

        ax = Axis(fig[3, 2], title=L"θ(n)", xlabel=L"n")
        lines!(ax, 0:Ncap(sm)-1, mf_phase, color = :red)
        lines!(ax, 0:Ncap(sm)-1, sm_phase, color = :blue)
        #lines!(ax, 0:length(C)-1, C, color = :red)
        xlims!(ax, 0,Ncap(sm)-1)
        ylims!(ax, 0, 2*pi)

        ax = Axis(fig[2:3, 3], title=L"K_t", xlabel=L"t")
        
        θr = range(0, 2*pi, length=100)
        lines!(ax, real(exp.(im.*θr)), imag(exp.(im.*θr)), color = :black, linewidth = 0.5,
            linestyle = :dash)
        scatter!(ax, sm_swkur, color = :black, alpha = 0.1, markersize = 6)
        scatter!(ax, mf_kur, color = :red,  markersize = 10)
        scatter!(ax, sm_kur, color = :blue, markersize = 10)
        
        xlims!(ax, -1,1)
        ylims!(ax, -1,1)

        ax = Axis(fig[2:3, 4], aspect=1, height=300, title=L"n_i")
        hidedecorations!(ax)
        hm = heatmap!(ax, img, colorrange = (0, 20),
            colormap = :RdBu)
        cm = Colorbar(fig[2:3, 5], hm)

        resize_to_layout!(fig)
    end
    fname = plotsdir("ising.gif")
    record(fig, fname, 1:tmax,
        framerate=20) do t
        # update
        parallel_update!(sm)
        step!(dm)
        img[] = n2img(sm, false)
        mf_dist[] = Vector(dm.x)
        sm_dist[] = n2N(sm)
        mf_traj[] = push!(mf_traj[], dm.x[1])
        sm_traj[] = push!(sm_traj[], s2a(sm.s))
        
        sm_ikur = kuramoto(sm)
        nphase = kuramoto_phases(sm)
        sw_phases = [exp(im*nphase[min(n+1, Q)]) for n in sm.n]
        sm_swkur[] = [Point2f(real(sp), imag(sp)) for sp in sw_phases]
        sm_kur[] = [(real(sm_ikur), imag(sm_ikur))]
        sm_phase[] = nphase

        phase = combined_map_phase(dm.x, dm.p)
        mf_phase[] = phase
        mf_ikur = dm.x'*exp.(im.*phase)
        mf_kur[] = [(real(mf_ikur), imag(mf_ikur))]
    end
    run(`open $fname`)
end