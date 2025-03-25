include(srcdir("auxiliary.jl"));

@testset "Exponential weights" begin
    # Test exponential weights
    Q = 20
    C = exponential_weights(Q, 0.1)
    @test length(C) == Q
    @test C[1] == 1
end

