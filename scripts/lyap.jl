using DrWatson
@quickactivate "BioIsing"

begin # load project environment
    include(srcdir("spin-model.jl"));
    include(srcdir("mean-field.jl"));
    include(srcdir("auxiliary.jl"));   
    using ProgressBars
    using CairoMakie
    using FFTW
end


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Refractive model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# MF: Lyap spectrum along β
begin
    J = 1
    R = 3
    θ = 0
    βs = range(0, 5, 31)
    ic = spike_ic_mf(R, 0)
    dx = MMatrix{R+1,R}(simplex_orthobasis(R+1))

    nequi, nmeas = 10000, 1000

    meas = zeros(length(βs), R);
    Threads.@threads for (i,) in ProgressBar(idx_combinations([βs]))
        dm = RefractiveIMF(ic, J, θ, βs[i])
        forward!(dm, nequi)
        L, conv_info = lyap_spectrum(dm, dx)
        meas[i,:] = L
    end
end
begin
    f = Figure()
    ax = Axis(f[1,1], xlabel=L"β", ylabel=L"λ")
    for i in 1:R
        scatter!(ax, βs, meas[:,i], label="λ$i")
    end
    axislegend(ax)
    save(plotsdir("refractive-mf-lyap-spetrum.pdf"), f)
    display(f)

end