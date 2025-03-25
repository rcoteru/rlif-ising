begin # imports
    using ProgressBars;
    using LinearAlgebra;
    using Statistics;
    using Zygote;
    using JLD;
end 
begin
    using Revise;
    push!(LOAD_PATH, pwd());
    using RefractiveIsing;
end

begin # set parameters
    J = 6
    R = 10
    H = 0
    npoints = 1000
    B = 2.5;

    equi_fancy = true
    equi_chk = 10000
    equi_tol = 1e-10
    equi_max = 400000

    equi_stp = 10000
    meas_stp = 10000

    fname = "data/poster/exampleP.jld"
end;

pars = RefracParams(J, H, B, R)
# Calculate fixed points
fpx = mf_fixed_point(pars)
res = eigen(Zygote.jacobian(p -> mf_step(p, pars), fpx)[1])

P0 = zeros(R,R)
for i in 1:R
    vec = real.(res.vectors[:,i])
    interm = vec .- minimum(vec)
    P0[i,:] = interm/sum(interm)
end

good_vecs = P0[1:2:end,:]
for _ in 1:100000
    for i in 1:1:Integer(R/2)
        good_vecs[i,:] = mf_step(good_vecs[i,:], pars)
    end
end

begin
    f = Figure(size=(1000,600))
    ax1 = Axis(f[1, 1])
    for i in 1:1:Integer(R/2)
        lines!(ax1, good_vecs[i,:])
    end
    f
end

vecs = zeros(2,R)
vecs[1,:] = good_vecs[3,:]
vecs[2,:] = good_vecs[4,:]

begin
    save(fname, "vecs", vecs)
end

vecs = load(fname, "vecs")

begin
    f = Figure(size=(500,300))
    ax1 = Axis(f[1, 1], xlabel="k", title=L"P^{(k)}",
        xticks=0:1:R-1)
    colors = [:red, :blue]
    for i in [1,2]
        lines!(ax1, 0:1:R-1, vecs[i,:], color=colors[i], label="$i-peak")
    end
    axislegend(ax1)
    f
end
save("figures/poster/exampleP.pdf", f)