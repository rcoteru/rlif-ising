begin # imports
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using Zygote;
    using JLD;
    using CairoMakie;
    using GLMakie;
end 
begin # long imports
    using Revise;
    push!(LOAD_PATH, pwd());
    using RefractiveIsing;
end

begin # set parameters
    J = [6]
    R = [5]
    npoints = [size(R)[1], size(J)[1], 301, 401]
    H = LinRange(-1.5, 1.5, npoints[3]);
    B = LinRange(0, 4, npoints[4]);

    equi_fancy = true
    equi_chk = 10000
    equi_tol = 1e-10
    equi_max = 500000

    equi_stp = 10000
    meas_stp = 30000

    fname = "data/article/phase_stats.jld";
end;

# begin # run the mf simulation
#     meas = zeros(4, npoints...)
#     it = collect(Base.product(1:npoints[1], 1:npoints[2], 1:npoints[3], 1:npoints[4]))
#     Threads.@threads for (i, j, k, l) in ProgressBar(it)
#         # define the system parameters
#         pars = RefracParams(J[j], H[k], B[l], R[i])
#         P    = mf_spike(pars)
#         # equilibration
#         if equi_fancy # fancy method with FFT
#             P = mf_equi_fft(P, pars, equi_chk, equi_tol, equi_max)
#         else # simple method
#             P = mf_equi(P, pars, equi_stp)
#         end
#         # measurement
#         traj = mf_meas(P, pars, meas_stp)
#         # stats
#         meas[1,i,j,k,l] = mean(traj[1,:])
#         meas[2,i,j,k,l] = std(traj[1,:])
#         meas[3,i,j,k,l] = mean(traj[4,:])
#         meas[4,i,j,k,l] = std(traj[4,:])
#     end
# end

# begin # save the data
#     save(fname, "meas", meas, "H", H, "B", B, "J", J, "R", R)
# end

begin # load the data
    meas  = load(fname, "meas")
end;


#meas
CairoMakie.activate!()
begin
    
    f = Figure(size=(700,350))
    # MF PLOTS

    ax1 = Axis(f[1,1], xlabel=L"\beta", title=L"\langle N_0 \rangle", ylabel=L"\theta",
    xticks = 0:1:9, yticks = -2:0.5:2, limits=(minimum(B), maximum(B), minimum(H), maximum(H)))
    heatmap!(ax1, B, H, meas[1,1,1,:,:]', colorrange=(0, 0.3),
    interpolate=true)
    #text!(7.5, -1.5, text = L"mean($m$)", align = (:center, :center), fontsize = fsize, color=:white)

    ax2 = Axis(f[1,2], xlabel=L"\beta", title=L"\sigma(N_0)",
    xticks = minimum(B)+1:1:maximum(B), yticks = minimum(H):0.5:maximum(H), 
    limits=(minimum(B), maximum(B), minimum(H), maximum(H)))
    hm = heatmap!(ax2, B, H, meas[2,1,1,:,:]', colorrange=(0, 0.3),
    interpolate=true)
    #text!(7.5, -1.5, text = L"std($m$)", align = (:center, :center), fontsize = fsize, color=:white)

    linkyaxes!(ax1, ax2)
    linkxaxes!(ax1, ax2)
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false
    Colorbar(f[1,3], hm, ticks=0:0.1:0.3)

    colgap!(f.layout, 5)

    f
end
save("figures/article/phase_stats.pdf", f)