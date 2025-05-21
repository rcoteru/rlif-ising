include(srcdir("mean-field.jl"));
include(srcdir("auxiliary.jl"));

# IC tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "ICs" begin

    # Quiet
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Quiet Vanilla IC
    x = quiet_ic_mf(0, 0)
    @test length(x) == 1
    @test x[1] == 0

    # Quiet Refractive IC
    x = quiet_ic_mf(5, 0)
    @test length(x) == 6
    @test sum(x) == 1
    @test x[6] == 1
    
    # Quiet Integrative IC
    x = quiet_ic_mf(0, 5)
    @test length(x) == 6
    @test sum(x) == 1
    @test x[6] == 1

    # Quiet Complete IC
    x = quiet_ic_mf(5, 5)
    @test length(x) == 11
    @test sum(x) == 1
    @test x[11] == 1

    # Spike
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Spike Vanilla IC
    x = spike_ic_mf(0, 0)
    @test length(x) == 1
    @test x[1] == 1

    # Spike Refractive IC
    x = spike_ic_mf(5, 0)
    @test length(x) == 6
    @test sum(x) == 1
    @test x[1] == 1

    # Spike Integrative IC
    x = spike_ic_mf(0, 5)
    @test length(x) == 6
    @test sum(x) == 1
    @test x[1] == 1

    # Spike Complete IC
    x = spike_ic_mf(5, 5)
    @test length(x) == 11
    @test sum(x) == 1
    @test x[1] == 1

    # Random
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Random Vanilla IC
    x = random_ic_mf(0, 0)
    @test length(x) == 1
    @test x[1] <= 1
    @test x[1] >= 0

    # Random Refractive IC
    x = random_ic_mf(5, 0)
    @test length(x) == 6
    @test sum(x) ≈ 1

    # Random Integrative IC
    x = random_ic_mf(0, 5)
    @test length(x) == 6
    @test sum(x) ≈ 1

    # Random Complete IC
    x = random_ic_mf(5, 5)
    @test length(x) == 11
    @test sum(x) ≈ 1

end

# VanillaIMF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Vanilla" begin
    # β = 0
    mf = VanillaIMF([1], 0, 0, 0, 0)
    step!(mf)
    @test mf.x0[1] == 1
    @test mf.x == [0.5]

    # β = 1, θ = 0, I = 0
    mf = VanillaIMF([1], 1, 0, 1, 0)
    step!(mf)
    @test mf.x == [0.5*(1+tanh(1))]

    # β = 1, θ = -1, I = 0
    mf = VanillaIMF([1], 1, -1, 1, 0)
    step!(mf)
    @test mf.x == [0.5*(1+tanh(2))]
end

# RefractiveIMF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Refractive Fixed Point" begin
    x = refractive_fxp(1, 2, 5, 0, 4)
    @test x[1] == x[2] == x[3] == x[4]
    @test x[5] == 1-sum(x[1:4])
    @test sum(x) ≈ 1
    mf = RefractiveIMF(x, 1, 2, 5,0)
    step!(mf)
    @test mf.x ≈ x
end 
@testset "Refractive IMF" begin

    # β = 0, θ = 0, I = 0
    ic = spike_ic_mf(4, 0)
    mf = RefractiveIMF(ic, 0, 0, 0, 0)
    step!(mf)
    @test mf.x0 == [1,0,0,0,0]
    @test mf.x == [0,1,0,0,0]
    [step!(mf) for i in 1:3]
    @test mf.x == [0,0,0,0,1]
    step!(mf)
    @test mf.x == [0.5,0,0,0,0.5]

    # β = 1, θ = 0, I = 0
    ic = quiet_ic_mf(4, 0)
    mf = RefractiveIMF(ic, 1, 0, 1, 0)
    step!(mf)
    @test mf.x ≈ [0.5,0,0,0,0.5]
    step!(mf)
    @test mf.x ≈ [0.5^2*(1+tanh(0.5)),0.5,0,0,0.5^2*(1-tanh(0.5))]
    
    # β = 1, θ = -1, I = 0
    ic = quiet_ic_mf(4, 0)
    mf = RefractiveIMF(ic, 1, -1, 1, 0)
    step!(mf)
    @test mf.x ≈ [0.5*(1+tanh(1)),0,0,0,0.5*(1-tanh(1))]
end
@testset "Refractive Entropy" begin
    J = 1
    θ = 0
    β = 6
    I = 0
    R = 4
    dm = RefractiveIMF(spike_ic_mf(R, 0), J, θ, β, I)

    # fdist tests
    meas_stp = 100
    fdist = refractive_fdist_traj!(dm, meas_stp)
    @test size(fdist) == (meas_stp, 2, R+1)
    @test all(sum(fdist, dims=3) .≈ 1)
    @test all(fdist[:,1,1] .== fdist[:,1,1])

    # S tests
    S = refractive_entropy!(dm, meas_stp)
    @test size(S) == (meas_stp, 2)
    @test all(S.>= 0)
    @test all(S.<= 1)
end


# IntegratorIMF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Integrator Fixed Point" begin    
    J = 0.2
    θ = 1
    β = 1
    Q = 6
    I = 0.1
    α = 0.1
    C = exponential_weights(Q, α)

    fxp = integrator_fxp(1, 0, 6, 0.1, C)
    @test length(fxp) == Q+1
    @test sum(fxp) ≈ 1
    mf = IntegratorIMF(fxp, 1, 0, 6, 0.1, C)
    step!(mf)
    @test mf.x ≈ fxp

    x = 0.8 # test value for probabilities
    currs = integrator_fxp_currents(x, J, θ, β, I, C)
    probs = integrator_fxp_probs(x, J, θ, β, I, C)
    @test probs[1,:] == 0.5 .+ 0.5*tanh.(currs)
    @test probs[2,:] == 0.5 .- 0.5*tanh.(currs)

    @test size(probs) == (2,Q)
    @test sum(probs, dims=1) ≈ ones(1,Q)
    @test probs[1,1] == 0.5 + tanh(β*(J*C[1]*x+C[1]*I-θ))/2
    @test probs[2,1] == 0.5 - tanh(β*(J*C[1]*x+C[1]*I-θ))/2
    @test probs[1,2] == 0.5 + tanh(
        β*(J*(C[2]*probs[2,1] + C[1])*x
        +(C[2]+C[1])*I-θ))/2
    @test probs[2,2] == 0.5 - tanh(
        β*(J*(C[2]*probs[2,1] + C[1])*x
        +(C[2]+C[1])*I-θ))/2
    @test probs[1,3] ≈ 0.5 + tanh(
        β*(J*(C[3]*probs[2,2]*probs[2,1] + C[2]*probs[2,1] + C[1])*x 
        + (C[3]+C[2]+C[1])*I-θ))/2
    @test probs[2,3] ≈ 0.5 - tanh(
        β*(J*(C[3]*probs[2,2]*probs[2,1] + C[2]*probs[2,1] + C[1])*x 
        + (C[3]+C[2]+C[1])*I-θ))/2
end
@testset "Integrator IMF" begin

    # β = 0, θ = 0, I = 0
    Q = 4
    x0, C = quiet_ic_mf(0, Q), ones(Q)
    dm = IntegratorIMF(x0, 1, 0, 0, 0, C)
    step!(dm)
    @test dm.x0 == x0
    @test dm.x == [0.5, 0, 0, 0, 0.5]
    step!(dm)
    @test dm.x == [0.5, 0.25, 0, 0, 0.25]

    # β = 1, θ = 0
    x0, C = quiet_ic_mf(0, Q), ones(Q)
    dm = IntegratorIMF(x0, 1, 0, 1, 0, C)
    step!(dm)
    @test dm.x == [0.5, 0, 0, 0, 0.5]
    step!(dm)
    @test dm.x == [
        (0.5+0.5*tanh(0.5)),
        0.5*(0.5-0.5*tanh(0.5)), 0, 0, 
        0.5*(0.5-0.5*tanh(0.5))]
    # do another step

    # TODO: β = 1, θ = 1

end
@testset "Integrator Entropy" begin

end

# J = 0.1
# θ = 1
# β = 5
# I = 0
# Q = 50
# α = 0.1
# C = exponential_weights(Q, α)
# dm = IntegratorIMF(spike_ic_mf(0, Q), J, θ, β, I, C)

# forward!(dm,1000)

# # fdist tests
# meas_stp = 100
# fdist = integrator_fdist_traj!(dm, meas_stp)
# @test size(fdist) == (meas_stp, 2, Q+1)
# @test all(sum(fdist, dims=3) .≈ 1)
# @test all(fdist[:,1,1] .== fdist[:,1,1])



# CombinedIMF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@testset "Combined Fixed Point" begin

    J = 0.1
    θ = 1
    β = 5
    Q = 50
    R = 20
    I = 0
    C = exponential_weights(Q, 0.1)

    #combined_fxp(J, θ, β, I, R, C)

    
end

J = 0.1
θ = 1
β = 5
Q = 50
R = 20
I = 0
C = exponential_weights(Q, 0.1)


@testset "Combined IMF" begin
   
    # β = 0, θ = 0, I = 0
    R, Q = 2, 3
    x0, C = spike_ic_mf(R, Q), ones(Q)
    dm = CombinedIMF(x0, 1, 0, 0, 0, C)
    step!(dm)
    @test dm.x0 == [1,0,0,0,0,0]
    @test dm.x == [0,1,0,0,0,0]
    step!(dm)
    @test dm.x == [0,0,1,0,0,0]
    step!(dm)
    @test dm.x == [0.5,0,0,0.5,0,0]
    step!(dm)
    @test dm.x == [0.25,0.5,0,0,0.25,0]

    # β = 1, θ = 0, I = 0
    R, Q = 2, 3
    x0, C = quiet_ic_mf(R, Q), ones(Q)
    dm = CombinedIMF(x0, 1, 0, 1, 0, C)
    step!(dm)
    @test dm.x == [0.5,0.0,0,0,0,0.5]
    step!(dm)
    @test dm.x == [0.5*(1+tanh(0.5))/2,
        0.5,0,0,0,0.5*(1-tanh(0.5))/2]


end