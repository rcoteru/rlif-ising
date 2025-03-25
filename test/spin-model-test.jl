include(srcdir("spin-model.jl"));
include(srcdir("auxiliary.jl"));

# IC/Auxiliary tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#TODO extend to auxiliary conversions
@testset "IC/Auxiliary" begin
    N = 100
    # Test spike initial conditions
    n = spike_ic_sm(N)
    @test length(n) == N
    @test all(n .== 0)
    
    # Test random initial conditions
    n = random_ic_sm(N, 20)
    @test length(n) == N
    @test all(n .>= 0)
    @test all(n .<= 20)
end

# Constructors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Constructors" begin

    N = 100
    Q = 20
    J = 1
    R = 5
    C = exponential_weights(Q, 0.1)
    β = 1
    θ = 1
    I = 0.1
    n = spike_ic_sm(N)

    # Vanilla constructor
    sm = VanillaSM(J, θ, β, I, n)
    @test sm.N == N
    @test sm.Q == 1
    @test sm.J == J
    @test sm.R == 0
    @test sm.C == [1]
    @test sm.s == ones(N)
    @test sm.n == n
    @test sm.a == [1]
    @test sm.θ == θ
    @test sm.β == β
    @test sm.I == I

    # Refractive constructor
    sm = RefractiveSM(J, θ, β, I, n, R)
    @test sm.N == N 
    @test sm.Q == 1
    @test sm.J == J
    @test sm.R == R
    @test sm.C == [1]
    @test sm.s == ones(N)
    @test sm.n == n
    @test sm.a == [1]
    @test sm.θ == θ
    @test sm.β == β
    @test sm.I == I

    # Integrator constructor
    sm = IntegratorSM(J, θ, β, I, n, C)
    @test sm.N == N
    @test sm.Q == Q
    @test sm.J == J
    @test sm.R == 0
    @test sm.C == C
    @test sm.s == ones(N)
    @test sm.n == n
    @test sm.a == [1, zeros(Q-1)...]
    @test sm.θ == θ
    @test sm.β == β
    @test sm.I == I

    sm = CompleteSM(J, θ, β, I, n, R, C)
    @test sm.N == N
    @test sm.Q == Q
    @test sm.J == J
    @test sm.R == R
    @test sm.C == C
    @test sm.s == ones(N)
    @test sm.n == n
    @test sm.a == [1, zeros(Q-1)...]
    @test sm.θ == θ
    @test sm.β == β
    @test sm.I == I

end

# Current/Probability Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Local field" begin
    
    J = 1
    N = 100
    Q = 20
    R = 5
    I = 0.1
    C = exponential_weights(Q, 0.1)
    n = spike_ic_sm(N)

    begin # β = 0, θ = 0
        sm = CompleteSM(J, 0, 0, I, n, R, C)
        @test local_current(sm, 1) == C[1]*(J+I)
        @test local_current(sm, 2) == C[1]*(J+I) + C[2]*I
        @test fprob(sm, 1) == 0.5
        fps = fprob(sm)
        @test all(fps .== 0.5)
        @test length(fps) == N
    end
    begin # β = 1, θ = 0
        sm = CompleteSM(J, 0, 1, I, n, R, C)
        @test fprob(sm, 1) == 0.5*(1+tanh(1.1))
        fps = fprob(sm)
        @test all(fps .==  0.5*(1+tanh(1.1)))
    end
    begin # β = 1, θ = 1
        sm = CompleteSM(J, 1, 1, I, n, R, C)
        @test fprob(sm, 1) == 0.5*(1+tanh(0.1))
        fps = fprob(sm)
        @test all(fps .==  0.5*(1+tanh(0.1)))
    end
    begin # β = 1, θ = 0, size 2 spike
        n = [zeros(Int(N/2))..., ones(Int(N/2))...]
        sm = sm = CompleteSM(J, 0, 1, I, n, R, C)
        @test local_current(sm, 1) == C[1]*(J*n2a(sm)[1]+I)
        @test local_current(sm, 2) == C[1]*(J*n2a(sm)[1]+I) + C[2]*(J*n2a(sm)[2]+I)
    end
end

# Parallel Update
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Parallel updates" begin
    
    N = 5
    Q = 20
    R = 5
    I = 0.1
    C = exponential_weights(Q, 0.1)
    n = spike_ic_sm(N)

    # β = 0
    sm = CompleteSM(1, 0, 0, I, n, R, C)
    @test all(rcheck(sm) .== false)
    @test all(fprob(sm) .== 0.5)
    parallel_update!(sm)
    @test all(sm.s .== -1)
    @test all(sm.n .== 1)

    # TODO: do some more tests

end

# Sequential Update
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Sequential updates" begin
    
    N = 5
    Q = 20
    J = 1
    R = 5
    I = 0.1
    C = exponential_weights(Q, 0.1)
    n = spike_ic_sm(N)
    sm = CompleteSM(J, 0, 1, I, n, R, C)

    # single spin updates
    # goal: do nothing
    sna_single_update(sm, 1, 1)
    @test sm.s[1] == 1
    @test sm.a[1] == 1
    @test sm.n[1] == 0
    # goal: switch off
    sna_single_update(sm, 1, -1)
    @test sm.s[1] == -1
    @test sm.a[1] == (N-1)/N
    @test sm.a[2] == 1/N
    @test sm.n[1] == 1
    # goal: switch on
    sna_single_update(sm, 1, 1)
    @test sm.s[1] == 1
    @test sm.a[1] == 1
    @test sm.n[1] == 0
    # goal: switch off when off
    sna_single_update(sm, 1, -1)
    sna_single_update(sm, 1, -1)
    @test sm.s[1] == -1
    @test sm.a[1] == (N-1)/N
    @test sm.a[2] == 0
    @test sm.a[3] == 1/N
    @test sm.n[1] == 2
    # goal: switch on again
    sna_single_update(sm, 1, 1)
    @test sm.s[1] == 1
    @test sm.a[1] == 1
    @test sm.a[2] == 0
    @test sm.a[3] == 0
    @test sm.n[1] == 0

    # test glauber step
    # should not fire bc n<R
    glauber_step!(sm, 1)
    @test sm.s[1] == -1
    @test sm.a[1] == (N-1)/N
    @test sm.a[2] == 1/N
    @test sm.n[1] == 1

    #TODO some more tests maybe?

    # test sequential_update
    sequential_update!(sm)
    @test sm.s[1] == -1
    @test sm.a[1] == 0
    @test sm.a[2] == (N-1)/N
    @test sm.n[1] == 2

end

# Trajectory Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Trajectories" begin
    #TODO
end