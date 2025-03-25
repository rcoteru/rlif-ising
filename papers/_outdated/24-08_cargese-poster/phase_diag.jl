begin # imports
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using Zygote;
    using JLD;
end 
begin # long imports
    using Revise;
    push!(LOAD_PATH, pwd());
    using RefractiveIsing;
end

begin # set parameters
    J = [2,6]
    R = [5,20]

    npoints = [size(R)[1], size(J)[1], 500, 500]
    H = LinRange(-2, 5, npoints[3]);
    B = LinRange(0, 10, npoints[4]);

    equi_fancy = true
    equi_chk = 10000
    equi_tol = 1e-10
    equi_max = 500000

    equi_stp = 10000
    meas_stp = 10000

    fname = "data/mf_poster-phase-uhr.jld";
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

# begin # run the eigenvalue calculation
#     thold = 1e-6
#     #vals = Dict(i => zeros(ComplexF64, (npoints[2:end]..., R[i])) for i in 1:npoints[1])
#     nstab = Dict(i => zeros((3, npoints[2:end]...)) for i in 1:npoints[1])
#     it = collect(Base.product(1:npoints[1], 1:npoints[2], 1:npoints[3], 1:npoints[4]))
#     Threads.@threads for (i, j, k, l) in ProgressBar(it)
#         # define the system parameters
#         pars = RefracParams(J[j], H[k], B[l], R[i])
#         # Calculate fixed point
#         fpx = mf_fixed_point(pars)
#         # Calculate Jacobian and diagonalize
#         res = eigen(Zygote.jacobian(p -> mf_step(p, pars), fpx)[1])
#         #vals[i][j,k,l,:] = res.values
#         nstab[i][1,j,k,l] = sum(abs.(res.values) .< (1-thold)) # small eigenvalues
#         nstab[i][3,j,k,l] = sum(abs.(res.values) .> (1+thold)) # large eigenvalues
#         nstab[i][2,j,k,l] = R[i] - nstab[i][1,j,k,l] - nstab[i][3,j,k,l] # intermediate eigenvalues
#     end
# end

# begin # save the data
#     save(fname, "meas", meas, "vals", vals, "nstab", nstab, 
#     "H", H, "B", B, "J", J, "R", R)
# end
# begin # save the data
#     save(fname, "nstab", nstab, "H", H, "B", B, "J", J, "R", R)
# end
begin # load the data
    #meas = load(fname, "meas")
    #vals = load(fname, "vals")
    nstab = load(fname, "nstab")
    H = load(fname, "H")
    B = load(fname, "B")
    J = load(fname, "J")
    R = load(fname, "R")
end;

begin
    # phase 1 - stable phase
    # phase 2 - oscillatory phase
    # phase 3 - echo phase
    phases = zeros(Int64, npoints...)
    zero_thold = 0.01
    for (i, j, k, l) in Iterators.product(axes(phases)...)
        # phase rules
        if nstab[i][1,j,k,l] == R[i]-1 # all unit eigenvalues 
            phases[i,j,k,l] = 1
        else
            if nstab[i][1,j,k,l] == 0 # at least one unit eigenvalue
                phases[i,j,k,l] = 3
            else
                phases[i,j,k,l] = 2
            end
        end
    end
end


using CairoMakie
begin
    Ridx = 1
    Jidx = 2
    f = Figure(size=(800,350))
    ax = Axis(f[1,1], xlabel=L"\beta", ylabel=L"\theta",
    xticks = 0:2:10, yticks = -1:1:5, limits=(0, 10, -2, 5))
    #heatmap!(B, H, phases[Ridx,Jidx,:,:]')

    Hpoints = size(phases)[3]
    Bpoints = size(phases)[4]
    # detect the left boundary
    boundary = zeros(Hpoints)
    for i in 1:Hpoints
        start = phases[Ridx,Jidx,i,1]
        for j in 1:Bpoints
            if phases[Ridx,Jidx,i,j] != start
                boundary[i] = B[j]
                break
            else
                boundary[i] = NaN
            end
        end
    end
    lines!(boundary, H, color=:black, linewidth=1)
    
    # detect the lower boundary
    boundary = zeros(Bpoints)
    for i in 1:Bpoints
        start = phases[Ridx,Jidx,1,i]
        for j in 1:Hpoints
            if phases[Ridx,Jidx,j,i] != start
                boundary[i] = H[j]
                break
            else
                boundary[i] = NaN
            end
        end
    end
    lines!(B[30:end], boundary[30:end], color=:black,linewidth=1)


    peak = minimum(filter(!isnan,boundary))
    println(peak)

    # detect the upper boundary
    boundary = zeros(Bpoints)
    for i in 1:Bpoints
        start = phases[Ridx,Jidx,Hpoints,i]
        if start == 3
            for j in Hpoints:-1:1
                if phases[Ridx,Jidx,j,i] != start
                    boundary[i] = H[j]
                    break
                end
            end
        else
            boundary[i] = NaN
        end
    end
    lines!(B, boundary, color=:black, linewidth=1, linestyle=:dash)


    # label the phases
    fsize = 25
    text!(6, -1.2, text = "1", align = (:center, :center), fontsize = fsize)
    text!(2, 1, text = "2a", align = (:center, :center), fontsize = fsize)
    text!(6, 3, text = "2b", align = (:center, :center), fontsize = fsize)


    # hide axis grid
    ax.xgridvisible = false
    ax.ygridvisible = false

    # hide axis ticks
    ax.xticksvisible = false
    ax.yticksvisible = false

    f
end
save("figures/poster/phase_diag.pdf", f)