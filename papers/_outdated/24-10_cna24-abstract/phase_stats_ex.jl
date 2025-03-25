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

begin # run the mf simulation
    meas = zeros(4, npoints...)
    it = collect(Base.product(1:npoints[1], 1:npoints[2], 1:npoints[3], 1:npoints[4]))
    Threads.@threads for (i, j, k, l) in ProgressBar(it)
        # define the system parameters
        pars = RefracParams(J[j], H[k], B[l], R[i])
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
        meas[1,i,j,k,l] = mean(traj[1,:])
        meas[2,i,j,k,l] = std(traj[1,:])
        meas[3,i,j,k,l] = mean(traj[4,:])
        meas[4,i,j,k,l] = std(traj[4,:])
    end
end

begin # save the data
    save(fname, "meas", meas, "H", H, "B", B, "J", J, "R", R)
end