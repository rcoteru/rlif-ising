begin # imports
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using JLD;
    using Zygote;
    using GLMakie;
    using CairoMakie;
end 
begin # long imports
    using Revise;
    push!(LOAD_PATH, pwd());
    using RefractiveIsing;
end

# using Random;
# Random.seed!(42);

begin # set parameters
    J = 6
    R = 5
    H = -1
    npoints = 600
    B = LinRange(0, 2, npoints);

    equi_fancy = true 
    equi_chk = 10000
    equi_tol = 1e-10
    equi_max = 400000

    equi_stp = 10000
    meas_stp = 20000

    fname = "data/article/statistics.jld";
end;

begin # run the simulation
    stats = zeros(4, npoints);
    trajs = zeros(2, npoints, meas_stp);
    vals = zeros(ComplexF64, (npoints, R))
    vecs = zeros(ComplexF64, (npoints, R, R))
    fpxs = zeros((npoints, R))
    nstab = zeros(npoints)
    Threads.@threads for k in ProgressBar(1:npoints)
        # define the system parameters
        pars = RefracParams(J, H, B[k], R)
        #P    = mf_random_point(pars)
        P    = mf_spike(pars)
        fxp  = mf_fixed_point(pars)
        # eigenvalues
        res = eigen(Zygote.jacobian(p -> mf_step(p, pars), fxp)[1])
        vals[k,:] = res.values
        vecs[k,:,:] = res.vectors
        nstab[k] = sum(abs.(res.values) .> 1.000000001)
        # equilibration
        if equi_fancy # fancy method with FFT
            P = mf_equi_fft(P, pars, equi_chk, equi_tol, equi_max)
        else # simple method
            P = mf_equi(P, pars, equi_stp)
        end
        # measurement
        traj = mf_meas(P, pars, meas_stp)
        # stats
        (stats[1,k], stats[2,k]) = (mean(traj[1,:]), std(traj[1,:]))
        (stats[3,k], stats[4,k]) = (mean(traj[4,:]), std(traj[4,:]))
        # trajectories
        (trajs[1,k,:], trajs[2,k,:])  = (traj[1,:], traj[4,:])
    end
end

GLMakie.activate!()
CairoMakie.activate!()
begin
    f = Figure(size=(1100,300))

    # Time series
    nshow = 200
    pshow = [50, 170, 400]
    pcols = [:blue, :red, :green]
    for (i,k) in enumerate(pshow)
        if i == 1
            ax = Axis(f[i, 0], ylabel=L"N_{0,t}", title="a) Trajectories",
            limits=(0, nshow, 0, 0.5))
            ax.xticklabelsvisible = false
            ax.xticksvisible = false
        end
        if i == 2
            ax = Axis(f[i, 0], ylabel=L"N_{0,t}",
            limits=(0, nshow, 0, 0.5), yticks=0:0.1:0.4)
            ax.xticklabelsvisible = false
            ax.xticksvisible = false
        end
        if i == 3
            ax = Axis(f[i, 0], xlabel="Timestep", ylabel=L"N_{0,t}",
            limits=(0, nshow, 0, 0.5), yticks=0:0.1:0.4, xticks=0:50:200)
        end
        lines!(ax, 1:nshow, trajs[1,k,1:nshow], color=pcols[i])
    end

    # Bifurcation diagram
    ax1 = Axis(f[1:3, 1], xlabel=L"\beta", ylabel=L"|N-N_{\text{fx}}|", title="b) Bifurcation Diagram")
    lastk = 200
    for k in 1:npoints
        scatter!(ax1, B[k]*ones(lastk+1), trajs[2,k,end-lastk:end], 
            color=:gray, markersize=0.5)
    end
    brkpnt = 575
    lines!(ax1, B[1:brkpnt], stats[3,1:brkpnt], color=:black)
    lines!(ax1, B[brkpnt+1:end], stats[3,brkpnt+1:end], color=:black)

    vlines!(ax1, B[pshow], color=pcols, linestyle=:dash)

    # Eigenvalue plot
    colors = [:green, :red, :purple, :green, :blue]
    ax2 = Axis(f[1:3, 2], xlabel=L"\beta", ylabel=L"|\lambda_i|", title="c) Eigenvalues")
    for r in 1:R
        lines!(ax2, B, abs.(vals[:,r]), color=colors[r], label = L"\lambda_%$r")
    end

    colgap!(f.layout,10)
    rowgap!(f.layout,5)

    f
end
save("figures/article/transitions.pdf", f)