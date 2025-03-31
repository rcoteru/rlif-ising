include(srcdir("spin-model.jl"));
include(srcdir("auxiliary.jl"));

# Initial Conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Initial Conditions" begin
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

# Conversions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@testset "Conversions" begin
    # s2a
    @test s2a(ones(Int, 10)) == 1
    @test s2a(-1*ones(Int, 10)) == 0
    @test s2a([1, -1, -1, -1, -1]) == 0.2

    # n2s
    n = [1, 1, 0, 0, 0]
    @test n2s(n) == [-1, -1, 1, 1, 1]

    # s2n
    s = [1, 1, -1, -1, 1]
    @test s2n(s) == [0, 0, 1, 1, 0]

    # S2n
    s = [
        [-1, 1, -1, -1, -1]';
        [-1, -1, 1, 1, -1]';
        [-1, -1, -1, 1, 1]'
        ]
    @test S2n(s, 10) == [10,2,1,0,0]
    @test S2n(s, 10, false) == [10,0,1,1,2]

    # n2s
    n = [1, 1, 2, 0, 0]
    @test n2s(n) == [-1, -1, -1, 1, 1]

    # n2a
    n = [1, 1, 2, 0, 0]
    @test n2a(n, 1) == [0.4]
    @test n2a(n, 3) == [0.4,0.4,0.2]
    @test n2a(n, 5) == [0.4,0.4,0.2,0.0,0.0]

    # n2N
    n = [1, 1, 2, 0, 0]
    @test n2N(n, 2) == [0.4, 0.6]
    @test n2N(n, 3) == [0.4, 0.4, 0.2]
    @test n2N(n, 4) == [0.4, 0.4, 0.2, 0]

    # s2N
    s = [
        [-1, 1, -1, -1, -1]';
        [-1, -1, 1, 1, -1]';
        [-1, -1, -1, 1, 1]'
        ]
    @test S2N(s, 3) == [0.4, 0.2, 0.4]
    @test S2N(s, 4) == [0.4, 0.2, 0.2, 0.2]
    @test S2N(s, 5) == [0.4, 0.2, 0.2, 0.0, 0.2]

    @test S2N(s, 3, false) == [0.2, 0.4, 0.4]
    @test S2N(s, 4, false) == [0.2, 0.4, 0.2, 0.2]
    @test S2N(s, 5, false) == [0.2, 0.4, 0.2, 0.0, 0.2]
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

    sm = CombinedSM(J, θ, β, I, n, R, C)
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

# Metrics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@testset "Metrics" begin
    
    N = 100
    Q = 20
    J = 1
    R = 5
    I = 0.1
    C = exponential_weights(Q, 0.1)
    n = spike_ic_sm(N)
    n2 = [zeros(20)..., ones(20)..., 
        2*ones(20)..., 30*ones(40)...]

    # Ncap
    sm = VanillaSM(J, 0, 0, I, n)
    @test Ncap(sm) == 1
    sm = RefractiveSM(J, 0, 0, I, n, R)
    @test Ncap(sm) == R+1
    sm = IntegratorSM(J, 0, 0, I, n, C)
    @test Ncap(sm) == Q+1
    sm = CombinedSM(J, 0, 0, I, n, R, C)
    @test Ncap(sm) == R+Q+1
    
    # n2a
    sm = CombinedSM(J, 0, 0, I, n, R, C)
    @test n2a(sm) == [1, zeros(Q-1)...] 
    @test n2a(sm.n, sm.Q) == [1, zeros(Q-1)...]
    sm = CombinedSM(J, 0, 0, I, n2, R, C)
    @test n2a(sm) == [0.2, 0.2, 0.2, zeros(Q-3)...]
    @test n2a(sm) == n2a(sm.n, sm.Q)

    # n2N
    sm = CombinedSM(J, 0, 0, I, n, R, C)
    @test n2N(sm) == [1, zeros(R+Q)...]
    @test n2N(sm.n, Ncap(sm)) == [1, zeros(R+Q)...]
    sm = CombinedSM(J, 0, 0, I, n2, R, C)
    @test n2N(sm) == [0.2, 0.2, 0.2, zeros(R+Q-3)..., 0.4]
    @test n2N(sm) == n2N(sm.n, Ncap(sm)) 

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
        sm = CombinedSM(J, 0, 0, I, n, R, C)
        @test local_current(sm, 1) == C[1]*(J+I)
        @test local_current(sm, 2) == C[1]*(J+I) + C[2]*I
        @test fprob(sm, 1) == 0.5
        fps = fprob(sm)
        @test all(fps .== 0.5)
        @test length(fps) == N
    end
    begin # β = 1, θ = 0
        sm = CombinedSM(J, 0, 1, I, n, R, C)
        @test fprob(sm, 1) == 0.5*(1+tanh(1.1))
        fps = fprob(sm)
        @test all(fps .==  0.5*(1+tanh(1.1)))
    end
    begin # β = 1, θ = 1
        sm = CombinedSM(J, 1, 1, I, n, R, C)
        @test fprob(sm, 1) == 0.5*(1+tanh(0.1))
        fps = fprob(sm)
        @test all(fps .==  0.5*(1+tanh(0.1)))
    end
    begin # β = 1, θ = 0, size 2 spike
        n = [zeros(Int(N/2))..., ones(Int(N/2))...]
        sm = sm = CombinedSM(J, 0, 1, I, n, R, C)
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
    sm = CombinedSM(1, 0, 0, I, n, R, C)
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
    sm = CombinedSM(J, 0, 1, I, n, R, C)

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