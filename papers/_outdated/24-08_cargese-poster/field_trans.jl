begin
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using FFTW;
    using JLD;
    using Zygote;

    using Revise;   
    push!(LOAD_PATH, pwd());
    using RefractiveIsing;
end

begin # set parameters
    J = 6
    R = 5
    B = 4
    npoints = 1500
    H = LinRange(-1, 2, npoints);

    equi_fancy = true 
    equi_chk = 10000
    equi_tol = 1e-10
    equi_max = 400000

    equi_stp = 10000
    meas_stp = 20000
    fname = "data/poster/field_trans.jld"
end;

# begin    
#     using Random;
#     Random.seed!(42);
# end

# begin     
#     using GLMakie
#     pars = RefracParams(0, 0, 0, R)
#     P0 = mf_random_point(pars)
# end
# begin
#     f = Figure(size=(1000,600))
#     ax = Axis(f[1, 1], xlabel="x", ylabel="y", title="Initial point")
#     lines!(ax, 1:R, P0, color=:red)
#     f
# end

# begin # run the simulation
#     stats = zeros(6, npoints);
#     trajs = zeros(3, npoints, meas_stp);
#     Threads.@threads for k in ProgressBar(1:npoints)
#         # define the system parameters
#         pars = RefracParams(J, H, B[k], R)
#         P    = copy(P0)
#         # P    = mf_random_point(pars)
#         # equilibration
#         if equi_fancy # fancy method with FFT
#             P = mf_equi_fft(P, pars, equi_chk, equi_tol, equi_max)
#         else # simple method
#             P = mf_equi(P, pars, equi_stp)
#         end
#         # measurement
#         traj = mf_meas(P, pars, meas_stp)
#         # stats
#         (stats[1,k], stats[2,k]) = (mean(traj[1,:]), std(traj[1,:]))
#         (stats[3,k], stats[4,k]) = (mean(traj[4,:]), std(traj[4,:]))
#         (stats[5,k], stats[6,k]) = (mean(traj[5,:]), std(traj[5,:]))
#         # trajectories
#         (trajs[1,k,:], trajs[2,k,:], trajs[3,k,:])  = (traj[1,:], traj[4,:], traj[5,:])
#     end
# end

begin # run the simulation
    vals = zeros(ComplexF64, (npoints, R))
    #vecs = zeros(ComplexF64, (npoints, R, R))
    fpxs = zeros((npoints, R))
    nstab = zeros(npoints)
    Threads.@threads for k in ProgressBar(1:npoints)
        # Define the system parameters
        pars = RefracParams(J, H[k], B, R)
        # Calculate fixed point
        fpx = mf_fixed_point(pars)
        # Calculate Jacobian and diagonalize
        res = eigen(Zygote.jacobian(p -> mf_step(p, pars), fpx)[1])
        vals[k,:] = res.values
        #vecs[k,:,:] = res.vectors
        nstab[k] = sum(abs.(res.values) .> 1.000000001)
    end
end;

# begin # save the data
#     data = save(fname, "stats", stats, "trajs", trajs, 
#     "H", H, "B", B, "J", J, "R", R)
# end;

begin # load the data
    stats = load(fname, "stats")
    trajs = load(fname, "trajs")
    H = load(fname, "H")
    B = load(fname, "B")
    J = load(fname, "J")
    R = load(fname, "R")
end;

using CairoMakie;

# begin
#     f = Figure(size=(1000,600))
#     ax1 = Axis(f[1, 1], xlabel="B", ylabel="Mean(fpd)", title="Bifurcation diagram")
#     lastk = 100
#     for k in 1:npoints
#         scatter!(ax1, H[k]*ones(lastk+1), trajs[2,k,end-lastk:end], color=:red)
#     end
#     lines!(ax1, H, stats[3,:], color=:blue)
#     lines!(ax1, H, stats[3,:]+stats[4,:], color=:green)
#     lines!(ax1, H, stats[3,:]-stats[4,:], color=:green)
#     f
# end

# begin
#     f = Figure(size=(1000,600))
#     x1 = Axis(f[1, 1])
#     lines!(x1, B, stats[3,:], color=:red)
#     f
# end

# begin
#     f = Figure(size=(1000,600))
#     k = 350
#     x1 = Axis(f[1, 1], title="Trajectory at theta = $(H[k])")
#     lines!(x1, 1:meas_stp, trajs[1,k,:], color=:red)
#     f
# end


begin # final figure
    f = Figure(size=(900,400), backgroundcolor = :white)

    # TRAJECTORY PLOT #
    nshow = 50
    points = [1200, 500, 350, 150]
    colors = [:gold, :green, :orangered, :magenta]
    ax1 = Axis(f[1:2, 1], ylabel=L"P^{(0)}", xlabel=L"t", xticks=0:10:40)
    for (i, k) in enumerate(points)
        lines!(ax1, 1:nshow, trajs[1,k,1:nshow], color=colors[i], label=" β = $(round(H[k], digits=1))")
    end
    axislegend(ax1, position=:rt, nbanks=2)
    ylims!(ax1, 0, 1)
    xlims!(ax1, 1, nshow)

    # FIXED POINT DISTANCE #
    lastk = 500
    nshow = 1500
    ax2 = Axis(f[1, 2:3], ylabel=L"|P-P_{\text{fx}}|", yaxisposition=:right, xticks=-0.8:0.2:1.6,
        yticks = 0.2:0.2:1, xticksvisible=false, xticklabelsvisible=false)
    for k in 1:nshow
        scatter!(ax2, H[k]*ones(lastk+1), trajs[2,k,end-lastk:end], color=:red,
            markersize=0.5, marker=:circle, strokewidth=0, rasterize=2)
    end
    lines!(ax2, H[1:nshow], stats[3,1:nshow], color=:blue)
    for (i, k) in enumerate(points)
        vlines!(ax2, H[k], 0, 1, color=colors[i], linestyle=:dash)
    end


    #band!(ax2, H[1:nshow], stats[3,1:nshow]-stats[4,1:nshow], 
    #    stats[3,1:nshow]+stats[4,1:nshow], color=:blue, alpha=0.2)
    ylims!(ax2, 0, 1)

    # FREQUENCY SPECTRUM #

    # cmap = :rainbow1
    # colors = cgrad(cmap, size(H)[1], categorical = true)
    # sg = f[2,1] = GridLayout()
    # img = zeros(npoints, Int(meas_stp//2)+1)
    # for (i, h) in enumerate(H)
    #     img[i,:] = abs.(rfft(trajs[1,i,:]))
    # end
    # ax = Axis(sg[1, 2], yticksvisible=false, yticklabelsvisible=false, 
    #     xlabel="Frequency", yaxisposition=:right, xticks=0:0.1:0.4)
    # for i in 1:5:npoints
    #     lines!(ax, rfftfreq(meas_stp)[2:end], img[i,2:end], color=colors[i])
    # end
    # xlims!(ax, 0, 0.5)
    # cb = Colorbar(sg[1, 1], limits = (minimum(H), maximum(H)), flipaxis = false,
    #     colormap=cmap, label=L"\theta", ticks=[-1,0,1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    cmap = :inferno
    img = zeros(npoints, Int(meas_stp//2))
    for (i, h) in enumerate(H)
        img[i,:] = sqrt.(abs.(rfft(trajs[1,i,:]))[2:end])
    end
    ax3 = Axis(f[2, 2:3], ylabel="Frequency (1/step)", xlabel=L"\theta", 
        yaxisposition=:right, xticks=-0.8:0.2:1.6)
    heatmap!(ax3, H, rfftfreq(meas_stp), img, colorscale=sqrt, colormap=cmap, rasterize=2)
    for (i, k) in enumerate(points)
        vlines!(ax3, H[k], 0, 1, color=colors[i], linestyle=:dash)
    end
    hlines!(ax3, [0.2, 0.4], -0.8, 1.6, color=:grey, linestyle=:dash)


    # EIGENVALUE PLOT #
    # ax4 = Axis(f[3, 2], yaxisposition=:right, ylabel=L"|\lambda|", xlabel=L"\theta")
    # for i in 1:2:R
    #     lines!(ax4, H[1:nshow], abs.(vals[1:nshow,i]), label="λ$i")
    # end

    linkxaxes!(ax2, ax3)
    #linkxaxes!(ax2, ax4)
    xlims!(ax3, -0.8, 1.6)
    #ylims!(ax4, 0.95, 1.05)

    rowgap!(f.layout, 5)
    colgap!(f.layout, 5)

    f
end
save("figures/poster/field_trans.pdf", f)
    
begin # old plot
    f = Figure(size=(1000,600))

    # mean activation plot
    # ax1 = Axis(f[1, 1], ylabel="Mean(P0)", title="Bifurcation diagram",
    #     xticksvisible=false, xticklabelsvisible=false)
    # lastk = 200
    # for k in 1:npoints
    #     scatter!(ax1, H[k]*ones(lastk+1), trajs[1,k,end-lastk:end], color=:red,
    #         markersize=1, marker=:circle, strokewidth=0)
    # end
    # lines!(ax1, H, stats[1,:], color=:blue)
    #band!(ax1, H, stats[1,:]-stats[2,:], stats[1,:]+stats[2,:], color=:blue, alpha=0.2)

    # fpd plot
    ax2 = Axis(f[1, 1], ylabel="Mean(fpd)",  xticks=-1:0.1:2)
    lastk = 200
    for k in 1:npoints
        scatter!(ax2, H[k]*ones(lastk+1), trajs[2,k,end-lastk:end], color=:red,
            markersize=1, marker=:circle, strokewidth=0)
    end
    lines!(ax2, H, stats[3,:], color=:blue)
    #band!(ax2, H, stats[3,:]-stats[4,:], stats[3,:]+stats[4,:], color=:blue, alpha=0.2)
    
    # frequency spectrum
    img = zeros(npoints, Int(meas_stp//2))
    for (i, h) in enumerate(H)
        img[i,:] = sqrt.(abs.(rfft(trajs[1,i,:]))[2:end])
    end
    ax3 = Axis(f[2, 1], xlabel="beta", ylabel="Frequency Spectrum", xticks=-1:0.1:2)
    heatmap!(ax3, H, rfftfreq(meas_stp), img, colorscale=sqrt, colormap=:inferno)

    #linkxaxes!(ax1, ax2)
    linkxaxes!(ax2, ax3)

    f
end
