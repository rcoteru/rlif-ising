begin # imports
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using JLD;
    using FFTW;
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
    Hfx = -1
    Bfx = 1.5
    npoints = 600
    B = LinRange(0.25, 2, npoints);
    H = LinRange(-1.5, 1.5, npoints);

    equi_fancy = true 
    equi_chk = 10000
    equi_tol = 1e-10
    equi_max = 400000

    equi_stp = 10000
    meas_stp = 20000

    fname = "data/article/spectra.jld";
end;

begin # run the simulation
    stats = zeros(2, 4, npoints);
    trajs = zeros(2, 2, npoints, meas_stp);
    Threads.@threads for k in ProgressBar(1:npoints)
        # define the system parameters
        pars = RefracParams(J, Hfx, B[k], R)
        #P    = mf_random_point(pars)
        P    = mf_spike(pars)
        # equilibration
        if equi_fancy # fancy method with FFT
            P = mf_equi_fft(P, pars, equi_chk, equi_tol, equi_max)
        else # simple method
            P = mf_equi(P, pars, equi_stp)
        end
        # measurement
        traj = mf_meas(P, pars, meas_stp)
        # stats
        (stats[1,1,k], stats[1,2,k]) = (mean(traj[1,:]), std(traj[1,:]))
        (stats[1,3,k], stats[1,4,k]) = (mean(traj[4,:]), std(traj[4,:]))
        # trajectories
        (trajs[1,1,k,:], trajs[1,2,k,:])  = (traj[1,:], traj[4,:])
    end
    Threads.@threads for k in ProgressBar(1:npoints)
        # define the system parameters
        pars = RefracParams(J, H[k], Bfx, R)
        #P    = mf_random_point(pars)
        P    = mf_spike(pars)
        # equilibration
        if equi_fancy # fancy method with FFT
            P = mf_equi_fft(P, pars, equi_chk, equi_tol, equi_max)
        else # simple method
            P = mf_equi(P, pars, equi_stp)
        end
        # measurement
        traj = mf_meas(P, pars, meas_stp)
        # stats
        (stats[2,1,k], stats[2,2,k]) = (mean(traj[1,:]), std(traj[1,:]))
        (stats[2,3,k], stats[2,4,k]) = (mean(traj[4,:]), std(traj[4,:]))
        # trajectories
        (trajs[2,1,k,:], trajs[2,2,k,:])  = (traj[1,:], traj[4,:])
    end
end

begin # save the data
    save(fname, "stats", stats, "trajs", trajs, "Hfx", Hfx, "Bfx", Bfx,
    "H", H, "B", B, "J", J, "R", R)
end

GLMakie.activate!()
CairoMakie.activate!()
begin
    f = Figure(size=(700,350))

    cmap = :binary
    crange = (0, 8)

    yaxis = rfftfreq(meas_stp)
    Label(f[1:2,0], "Frequency", fontsize=16, rotation=pi/2)


    ax1 = Axis(f[1, 1], xlabel=L"\beta", 
        xticks=minimum(B):0.25:maximum(B), 
        yticks=minimum(yaxis):0.1:maximum(yaxis),
        xaxisposition=:top)
    img1 = zeros(npoints, Int(meas_stp//2))
    for (i, h) in enumerate(B)
        img1[i,:] = abs.(rfft(trajs[1,1,i,:]))[2:end]
    end
    heatmap!(ax1, B, yaxis, img1, colormap=cmap,
        rasterize=2, colorrange=crange)
    
    ax2 = Axis(f[2, 1], xlabel=L"\theta", 
        xticks=minimum(H):0.5:maximum(H),
        yticks=minimum(yaxis):0.1:maximum(yaxis)-0.1)
    img2 = zeros(npoints, Int(meas_stp//2))
    for (i, h) in enumerate(H)
        img2[i,:] = abs.(rfft(trajs[2,1,i,:]))[2:end]
    end
    hm = heatmap!(ax2, H, yaxis, img2, colormap=cmap, 
        rasterize=2, colorrange=crange)

    hlines!(ax1, [0.2,0.4], color=:gray, linestyle=:dash)
    hlines!(ax2, [0.2,0.4], color=:gray, linestyle=:dash)
    rowgap!(f.layout, 5)
    f
end
save("figures/article/spectra.pdf", f)